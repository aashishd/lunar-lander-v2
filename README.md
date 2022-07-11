Project 2 readme

This repository mainly consists of following parts:

# Lunar Lander Agent
lunar_lander_learner.py contains LunarLanderAgent which defines how the lunar lander agent is trained and
It mainly has two functions that we should know of :

1. __init__(self, gamma, learning_rate, batch_size, c_param=1)
This can used to initialize the LunarLanderAgent 
   We need to pass three mandatory parameters learning rate, gamma and batch_size
   Additionaly, we can choose to pass a c value, which decides the frequency of copying weights from old network to the new network.
   c_value signifies the number of epochs
   
2. train_network() :
   This function can be called directly without any parameters once the agent is initialized.
   It will train the 2 layered neural network with the parameters pass in constructor.
   Once the cumulative reward over 100 episodes becomes more than 200 or we reach 1500 episodes, the agnet terminates and saves a log file. resources folder.
   This log file can be used to plot various graphs.
   


# config.json
{
  "visualize": false,
  "stop_training": false,
  "print_params": false,
  "save_ckpt": true,
  "save_logs": true
}
This is a json file which consists of some of the parameters which we can use to to control the netowrk which it is training.
This is mainly used in order to perform some actions while the agent is training.


# graphs

This directly consists of all the graphs used in the report

# train_lunar_lander.py q
This consists of default parameters which worked well in my case and can be executed directly to train the DQN for lunar lander.

# train_lunar_analysis.py
This python file was contains code mainly used to genrate the graphs

   