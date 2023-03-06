import os
import sys
import json


def get_exp_config(file_name):
    if os.path.exists('JSP_env/config/' + file_name + '.json'):
        config = json.loads(open('JSP_env/config/' + file_name + '.json').read())
    else:
        config = {}
        """Flags to indicate whether certain task should be done"""
        config.update({'LOGGER_FLAG': True})  # Use Stable Baseline3 Logger
        config.update({'LOAD_FLAG': False})  # Load pre-existing model
        config.update({'SAVE_FLAG': False})  # Save model
        config.update({'RENDER_FLAG': False})  # Render Environment # TODO: Not implemented yet
        config.update({'MULT_ENV_FLAG': False})  # Use multiple stacked environments -> Use DummyVecEnv
        config.update({'EVAL_FLAG': True})  # Evaluate Model

        """Path from/to where files loaded/saved"""
        config.update({'logging_path': "JSP_env/log/"})  # Save log files here
        config.update({'save_path': 'JSP_env/models/'})  # Save models here

        """Configure Experiment"""
        config.update({'model_type': 'PPO'})  # Type of Model used in experiment
        if config['LOAD_FLAG']:
            config.update({'model_version': 'tbd.'})
            config.update({'load_path': config['save_path'] + config['model_type'] + '/' + config['model_version']})
        else:
            config.update({'load_path': ''})

        """Write to .JSON"""
        file_name = 'JSP_env/config/' + file_name[1:-2] + '.json'
        with open(file_name, 'w') as fp:
            json.dump(config, fp)

    return config


def get_env_config(file_name):
    if os.path.exists('JSP_env/config/' + sys.argv[2] + '.json'):
        parameters = json.loads(open('JSP_env/config/' + file_name + '.json').read())
        parameters.update({'max_episode_timesteps': 1_000})
    else:
        parameters = {}
        parameters.update({'max_episode_timesteps': 1_000})  # Save model

    return parameters
