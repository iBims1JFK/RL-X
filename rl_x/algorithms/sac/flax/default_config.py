from ml_collections import config_dict
import time


def get_config(algorithm, environment):
    config = config_dict.ConfigDict()

    config.algorithm_name = algorithm.name

    config.device = "cuda"  # cpu, cuda
    config.total_timesteps = 1e9
    config.nr_envs = 2
    config.learning_rate = 3e-4
    config.buffer_size = 1e6
    config.learning_starts = 5000
    config.batch_size = 2048
    config.tau = 0.005
    config.gamma = 0.99
    config.train_freq = 1
    config.gradient_steps = 1
    config.ent_coef = "auto"
    config.target_update_interval = 1
    config.target_entropy = "auto"

    return config
