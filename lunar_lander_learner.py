import json
import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class LunarLanderAgent(object):
    def __init__(self, gamma, learning_rate, batch_size, hidden_layers=[100, 50], c_param=1):
        # hyper parameters
        self.gamma = gamma
        self.init_epsilon = 1.0
        self.final_epsilon = 0.1
        self.epsilon_decay_rate = 0.01
        self.learning_rate = learning_rate
        self.c_param = c_param
        self.max_history = 200000
        self.hidden_layers = hidden_layers

        self.n_episodes = 2000
        self.max_time_steps = 400
        self.replay_batch = batch_size

        # environment
        self.env = gym.make("LunarLander-v2").env
        self.state_count = 8
        self.action_count = 4

        # DQN network
        self.dqn_network: Model = None
        self.dqn_network_target: Model = None

        # housekeeping
        self.optimizer = None
        #
        self.training_id = f"ap-new_2_layer_relu_g{gamma}-b{batch_size}-lr{learning_rate}-c{c_param}"
        self.experience_replay_queue = deque(maxlen=self.max_history)
        self.episode_count = 0
        self.episode_rewards = []
        self.train_log = {'batch_size': self.replay_batch, 'gamma': self.gamma,
                          'init_e': self.init_epsilon, 'final_e': self.final_epsilon,
                          'e_decay': 'linear_decay_0.01_per_10_steps',
                          'c_param': self.c_param,
                          'learning_rate': self.learning_rate,
                          'max_time_steps_per_episode': self.max_time_steps,
                          'deque': self.max_history,
                          'details': '',
                          'rewards_per_episode': []}
        self.stop_training = False
        self.visualize = False
        self.print_params = False
        self.save_ckpt = False
        self.save_logs = False
        self.restart_step = 901
        self.restart_training = False
        self.last_100_rewards = deque(maxlen=100)
        self.model_path = os.path.join("checkpoints", f"model_{self.training_id}")
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)

    def reload_config(self):
        if self.episode_count % 10 == 0:
            with open('config.json', 'r') as f:
                config_json = json.loads(f.read())
                self.visualize = config_json['visualize']
                self.stop_training = config_json['stop_training']
                self.print_params = config_json['print_params']
                self.save_logs = config_json['save_logs']
                self.save_ckpt = config_json['save_ckpt']

    def get_current_learning_rate(self):
        return float(K.eval(self.dqn_network.optimizer.lr))

    def save_training_log(self):
        self.train_log['rewards_per_episode'] = self.episode_rewards
        with open(f'resources/{self.training_id}.json', 'w') as f:
            f.write(json.dumps(self.train_log, indent=4, sort_keys=True))

    def load_training_log(self):
        # self.train_log['rewards_per_episode'] = self.episode_rewards
        with open(f'resources/{self.training_id}.json', 'r') as f:
            self.train_log = json.loads(f.read())
            self.episode_rewards = self.train_log['rewards_per_episode'][:self.episode_count]

    def solve(self):
        self.dqn_network = self.train_network()

    def copy_weights(self, source: Model, target: Model):
        """
        Copies the weights from backed up Q to new Q
        :param source:
        :param target:
        :return:
        """
        target.set_weights(source.get_weights())

    def create_dqn_network(self) -> Model:
        model = Sequential()
        # lr_schedule = ExponentialDecay(self.learning_rate,
        #                                decay_steps=10000,
        #                                decay_rate=0.96,
        #                                staircase=True)
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=self.state_count))
        for nodes in self.hidden_layers[1:]:
            model.add(Dense(nodes, activation='relu'))
        model.add(Dense(self.action_count, activation='linear'))

        # optimizer = RMSprop(
        #     learning_rate=self.learning_rate,
        #     rho=0.9,
        #     momentum=0.95,
        #     epsilon=1e-07,
        #     centered=False,
        #     name="RMSprop"
        # )

        # Create a callback that saves the model's weights
        # cp_callback = ModelCheckpoint(filepath=checkpoint_path,
        #                               save_weights_only=True,
        #                               verbose=1)

        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def get_epsilon_value(self):
        # if self.episode_count > 50:
        # decayed_epsilon = round(self.init_epsilon * (self.epsilon_decay_rate ** ((self.episode_count - 50) / 10)),2)
        decayed_epsilon = self.init_epsilon - int(self.episode_count / 10) * self.epsilon_decay_rate
        return max(decayed_epsilon, self.final_epsilon)
        # return self.init_epsilon

    def save_weights(self):
        if self.save_ckpt:
            if self.episode_count % 100 == 1:
                ckpt_path = os.path.join(self.model_path, f"{self.episode_count}.h5")
                self.dqn_network_target.save(ckpt_path)

    def train_network(self):

        total_terminal_steps = 0
        # init a Q-table with zero values
        self.dqn_network = self.create_dqn_network()
        self.dqn_network_target = self.create_dqn_network()
        if self.restart_training:
            self.episode_count += self.restart_step
            ckpt_path = os.path.join(self.model_path, f"{self.episode_count}.h5")
            # self.dqn_network_target.save(ckpt_path)
            self.dqn_network.load_weights(ckpt_path)
            self.load_training_log()
        self.copy_weights(self.dqn_network, self.dqn_network_target)

        print("Q n/w summary")
        self.dqn_network.summary()
        print("Target n/w summary")
        self.dqn_network_target.summary()
        # copy weights to the new Q network
        # self.copy_weights(self.dqn_network, self.dqn_network_target)
        # for i_episode in range(self.n_episodes):
        # PROCESS EVERY EPOCH
        while not self.stop_training:
            state = self.env.reset()  # init the state S
            # Q_table = np.copy(self.dqn_network)  # copy Q_table
            t = 0
            tot_rewards = 0
            self.reload_config()
            if self.print_params:
                print(f"Parameters are = {self.train_log}")
                self.print_params = False
            self.episode_count += 1
            # print(f"Curent-LR={self.get_current_learning_rate()}")
            # PROCESS EVERY TIME STEP
            while True:
                if self.visualize:
                    self.env.render()
                action = self.get_greedy_action(state)  # select take and action A using e-greedy

                next_state, reward, is_terminated, _ = self.env.step(action)
                tot_rewards += reward

                # store in experience replay queue

                t += 1  # update time step count
                if t > self.max_time_steps:
                    # reset self.Q
                    print(f"Max TS reached, Episode count {self.episode_count}")
                    is_terminated = True

                    # break

                self.experience_replay_queue.append([state, action, reward, next_state, is_terminated])

                if len(self.experience_replay_queue) > self.replay_batch:
                    self.experience_replay()

                state = next_state  # update the state state

                if is_terminated:
                    # print("Episode finished after {} timesteps".format(t + 1))
                    total_terminal_steps += t
                    # self.episode_count += 1

                    # update the log details
                    self.episode_rewards.append([self.episode_count, self.get_epsilon_value(), tot_rewards, t])
                    self.last_100_rewards.append(tot_rewards)
                    print(
                        f"Total reward for episode {self.episode_count} and time_steps = {t}, epsilon {self.get_epsilon_value()}"
                        f" = {tot_rewards}")

                    mean_rewards_last_100_episodes = sum(self.last_100_rewards) / len(self.last_100_rewards)
                    if mean_rewards_last_100_episodes > 200:
                        self.stop_training = True
                        print(
                            f"Mean reward for last {len(self.last_100_rewards)} steps = {mean_rewards_last_100_episodes}")
                    if self.episode_count % 20 == 0:
                        print(
                            f"Mean reward for last {len(self.last_100_rewards)} steps = {mean_rewards_last_100_episodes}")
                        if self.save_logs:
                            self.save_training_log()

                    if self.episode_count % self.c_param == 0:
                        print(f"Copying weights per {self.c_param} episodes now !!")
                        self.copy_weights(self.dqn_network, self.dqn_network_target)

                    self.save_weights()
                    break

            if self.episode_count > self.n_episodes:
                print("Maximum episode length reached !")
                break
        print(f"EpisodeRewards = {self.episode_rewards}")
        self.save_training_log()
        self.env.close()
        print(f"Avg timesteps = {total_terminal_steps / self.n_episodes}")
        return self.dqn_network

    def experience_replay(self):

        # replay the existing training data
        batch = random.sample(self.experience_replay_queue, self.replay_batch)
        input_states = np.array([tup[0] for tup in batch])
        next_states = np.array([tup[3] for tup in batch])
        predicted_outputs = self.dqn_network.predict(input_states)
        next_predictions = self.dqn_network_target.predict(next_states)

        rewards = np.array([tup[2] for tup in batch])
        updated_predictions = rewards + self.gamma * np.amax(next_predictions, axis=1)

        is_terminal = [tup[4] for tup in batch]
        actions = [tup[1] for tup in batch]

        y_hat = np.copy(predicted_outputs)

        for idx in range(self.replay_batch):
            cur_action = actions[idx]
            cur_reward = rewards[idx]
            if is_terminal[idx]:
                y_hat[idx][cur_action] = cur_reward
            else:
                # do something else
                y_hat[idx][cur_action] = updated_predictions[idx]

        # self.dqn_network.train_on_batch(input_states, y_hat)
        self.dqn_network.fit(input_states, y_hat, epochs=1, verbose=False)

    def get_greedy_action(self, state):
        if np.random.random() <= self.get_epsilon_value():
            # we chose randomly
            return np.random.randint(0, 4)
        else:
            # we exploit
            return np.argmax(self.dqn_network.predict(np.array([state]))[0])
