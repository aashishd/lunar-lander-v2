from lunar_lander_learner import LunarLanderAgent

if __name__ == '__main__':
    gammas = [0.99]
    learning_rates = [0.001]
    batch_sizes = [16]
    for _learning_rate in learning_rates:
        for _batch_size in batch_sizes:
            for _gamma in gammas:
                agent = LunarLanderAgent(_gamma, _learning_rate, _batch_size, hidden_layers=[50, 50, 50], c_param=2)
                agent.train_network()
