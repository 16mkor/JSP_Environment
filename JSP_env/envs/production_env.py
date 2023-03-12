import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten_space
import numpy as np
from datetime import datetime

from JSP_env.envs.initialize_env import *
from JSP_env.envs.initialize_env import _get_criteria_events
from JSP_env.envs.time_calc import Time_calc
from JSP_env.envs.logger import *


class ProductionEnv(gym.Env):

    def __init__(self, parameters, **kwargs):
        super(ProductionEnv, self).__init__(**kwargs)

        """Counter"""
        self.count_episode = 0
        self.count_steps = 0

        """Create Simpy Environment"""
        self.env = simpy.Environment()

        """Parameter settings of environment & agent are defined here"""
        # Setup parameter configuration
        if len(parameters) == 2:
            print('No Configuration provided!')
            self.parameters = define_production_parameters(env=self.env, episode=self.count_episode)
        else:
            print('Configuration provided!')
            self.parameters = parameters
            self.parameters = _get_criteria_events(env=self.env, parameters=parameters)
        # Dict of Agents
        self.agents = {}

        """Statistic parameter"""
        self.last_export_time = 0.0
        self.last_export_real_time = datetime.now()
        self.statistics, self.stat_episode = define_production_statistics(self.parameters)
        self.statistics['sim_start_time'] = datetime.now()

        """Initialize production system"""
        self.time_calc = Time_calc(parameters=self.parameters, episode=self.count_episode)
        self.resources = define_production_resources(env=self.env, statistics=self.statistics,
                                                     parameters=self.parameters, agents=self.agents,
                                                     time_calc=self.time_calc)

        """Observation and action space"""
        self.observation_space = flatten_space(spaces.Box(low=-1, high=100, shape=(self._state_size(),), dtype=np.float64))
        self.action_space = spaces.Discrete(self._action_size())

    def step(self, actions):
        truncated = False
        info = {}
        self.count_steps += 1

        if (self.step_counter % self.parameters['EXPORT_FREQUENCY'] == 0
            or self.step_counter % self.max_episode_timesteps == 0) \
                and not self.parameters['EXPORT_NO_LOGS']:
            self.export_statistics(self.step_counter, self.count_episode)

        if self.step_counter == self.parameters['max_episode_timesteps']:
            truncated = True

        # If multiple transport agents then for loop required
        for _ in Transport.agents_waiting_for_action:
            agent = Transport.agents_waiting_for_action.pop(0)
            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                agent.next_action = [int(actions)]
            elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
                agent.next_action = [int(actions[0]), int(actions[1])]
            agent.state_before = None

            self.parameters['continue_criteria'].succeed()
            self.parameters['continue_criteria'] = self.env.event()

            self.env.run(until=self.parameters[
                'step_criteria'])  # Waiting until action is processed in simulation environment
            # Simulation is now in state after action processing

            reward, terminal = agent.calculate_reward(actions)

            if terminal:
                print("Last episode action ", datetime.now())
                self._export_statistics(self.count_steps, self.count_episode)

            agent = Transport.agents_waiting_for_action[0]
            states = agent.calculate_state()  # Calculate state for next action determination

            if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions)]
            elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
                self.statistics['stat_agent_reward'][-1][3] = [int(actions[0]), int(actions[1])]
            self.statistics['stat_agent_reward'][-1][4] = round(reward, 5)
            self.statistics['stat_agent_reward'][-1][5] = agent.next_action_valid
            self.statistics['stat_agent_reward'].append(
                [self.count_episode, self.count_steps, round(self.env.now, 5),
                 None, None, None, states])

            return states, reward, terminal, truncated, info

    def reset(self):
        print("####### Reset Environment #######")
        print("Sim start time: ", self.statistics['sim_start_time'])

        """Reset counter"""
        self.count_episode += 1  # increase episode by one
        self.count_steps = 0  # reset the counter of timesteps

        """Change parameter to new szenario"""
        if self.count_episode == self.parameters['CHANGE_SCENARIO_AFTER_EPISODES']:
            self._change_production_parameters()

        # Setup and start simulation
        if self.env.now == 0.0:
            print('Run machine shop simpy environment')
            self.env.run(until=self.parameters['step_criteria'])

        obs = np.array(self.resources['transps'][0].calculate_state())
        info = {}
        return obs, info

    def close(self):
        print("####### Close Environment #######")
        """Export statistics of closed environment"""
        if not self.parameters['EXPORT_NO_LOGS']:
            self.statistics.update({'time_end': self.env.now})
            export_statistics_logging(statistics=self.statistics, parameters=self.parameters, resources=self.resources)
        super().close()

    def render(self, mode='human', close=False):
        print("####### Render Environment #######")
        pass

    def _state_size(self):
        state_type = 'bool'
        number = 0
        # Avaliable Action are always part of state vector
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            number = len(self.resources['transps'][0].mapping)
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            number = (len(self.resources['transps'][0].mapping) - 1) ** 2 + 1
        # State value alternatives sorted according to the type
        if 'bin_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'bin_location' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES'] + self.parameters['NUM_SINKS']
        if 'bin_machine_failure' in self.parameters['TRANSP_AGENT_STATE']:
            number += self.parameters['NUM_MACHINES']
        if 'int_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'int'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'rel_buffer_fill' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'rel_buffer_fill_in_out' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] * 2 + self.parameters['NUM_SOURCES']
        if 'order_waiting_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'order_waiting_time_normalized' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'distance_to_action' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES'] + self.parameters['NUM_SOURCES']
        if 'remaining_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES']
        if 'total_process_time' in self.parameters['TRANSP_AGENT_STATE']:
            state_type = 'float'
            number += self.parameters['NUM_MACHINES']

        print("State space size: ", number)
        return number  # dict(type=state_type, shape=(number))

    def _action_size(self):
        if self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'direct':
            number = len(self.resources['transps'][0].mapping)
        elif self.parameters['TRANSP_AGENT_ACTION_MAPPING'] == 'resource':
            number = len(self.resources['transps'][0].mapping)
            # return dict(type='int', num_values=number, shape=(2,))
        print("Action space size: ", number)
        return number  # dict(type='int', num_values=number)

    def _change_production_parameters(self):
        print("CHANGE_OF_PRODUCTION_PARAMETERS")
        pass

    def _export_statistics(self, counter, episode_counter):
        pass
