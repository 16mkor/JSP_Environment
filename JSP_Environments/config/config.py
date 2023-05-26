import json
import os


def get_settings(file_name, model_type):
    if os.path.exists('JSP_Environments/config/' + file_name + '.json'):
        config = json.loads(open('JSP_Environments/config/' + file_name + '.json').read())
    else:
        config = {}
        """Flags to indicate whether certain task should be done"""
        config.update({'LOGGER_FLAG': False})  # Use Stable Baseline3 Logger
        config.update({'LOAD_FLAG': False})  # Load pre-existing model
        config.update({'SAVE_FLAG': True})  # Save model
        config.update({'RENDER_FLAG': False})  # Render Environment # TODO: Not implemented yet
        config.update({'MULT_ENV_FLAG': False})  # Use multiple stacked environments -> Use DummyVecEnv
        config.update({'EVAL_FLAG': True})  # Evaluate Model
        config.update({'HYPERPARAM_FLAG': True})  # Do Hyperparameter tuning

        """Path from/to where files loaded/saved"""
        config.update({'logging_path': "JSP_Environments/log/"})  # Save log files here
        config.update({'save_path': 'JSP_Environments/models/'})  # Save models here
        if not os.path.exists(config['save_path']):
            os.makedirs(config['save_path'])

        """Configure Experiment"""
        config.update({'model_type': model_type})  # Type of Model used in experiment
        if config['LOAD_FLAG']:
            config.update({'model_version': 'tbd.'})  # TODO: Needs to be defined, if needed
            config.update({'load_path': config['save_path'] + config['model_type'] + '/' + config['model_version']})
        else:
            config.update({'load_path': ''})

        """To use Tensorboard, a log location for the RL agent is needed"""
        config.update(
            {'tensorboard_log': 'JSP_Environments/log/'})  # /tensorboard_' + config['model_type'] + '_' + time + '/'})
        if not os.path.exists(config['tensorboard_log']):
            os.makedirs(config['tensorboard_log'])
        print('Tensorboard command: tensorboard --logdir ' + config['tensorboard_log'])

        """Write to .JSON"""
        # file_name = 'JSP_Environments/config/' + file_name[1:-2] + '.json'
        # with open(file_name, 'w') as fp:
        #     json.dump(config, fp)

    return config


def get_env_config(file_name):
    if os.path.exists('JSP_Environments/config/' + file_name + '.json'):
        parameters = json.loads(open('JSP_Environments/config/' + file_name + '.json').read())
    else:
        parameters = 'NO_PARAMETERS'
    return parameters
