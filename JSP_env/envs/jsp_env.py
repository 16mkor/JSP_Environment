import gym
from gym import spaces
import pygame
import numpy as np
import datetime as dt
import random
import json
import os
from JSP_env.envs.initialize_env import *


class JSPEnv(gym.Env):
    # Global parameters
    PRINT_CONSOLE = False  # Extended print out during running, particularly for debugging
    EPSILON = 0.000001  # Small number larger than zero used as "marginal" time step or to compare values
    EXPORT_FREQUENCY = 10 ** 3  # Number of steps between csv-export of log-files
    EXPORT_NO_LOGS = False  # Turn on/off export of log-files
    PATH_TIME = "log/" + dt.now().strftime("%Y%m%d_%H%M%S")

    def __init__(self, config_file: str = "config_ppo.json", config_stats: str = "config_stats.json"):

        # Inherit from super class
        super(JSPEnv, self).__init__()

        # Setup parameter configuration
        if os.path.exists('JSP_env/config/' + config_file):
            self.parameter = json.load('JSP_env/config/' + config_file)
        else:
            self.parameter = define_production_parameters()

        # Setup statistics
        if os.path.exists('JSP_env/config' + config_stats):
            self.statistics = json.load('JSP_env/config' + config_stats)
            self.stat_episode = {}
            for stat in statistics['episode_statistics']:
                self.stat_episode.update({stat: np.array([0.0] * len(statistics[stat]))})
        else:
            self.statistics, self.stat_episode = define_production_statistics(self.parameter)

        # Initialize production resources
        self.resoucres = define_production_resources()

        self.observation_space = spaces.Dict()
        self.action_space = spaces.Discrete()
        self.window = None
        self.clock = None

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self, mode: str = "rgb_array"):
        pass

    def _render_frame(self):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
