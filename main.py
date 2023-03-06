from JSP_env.run import run

if __name__ == "__main__":
    LOGGER_FLAG = True
    logging_path = "JSP_env/log/"
    LOAD_FLAG = False
    model_version = None
    SAVE_FLAG = False
    RENDER_FLAG = False
    MULTENV_FLAG = False
    EVAL_FLAG = False
    save_path = 'JSP_env/models/'
    model_type = 'PPO'
    load_path = save_path + model_type + '/' + model_version

    # Run the experiments with a certain configuration
    run(model_type=model_type, load_path=load_path,
        save_path=save_path, logging_path=logging_path,
        LOAD_FLAG=LOAD_FLAG, LOGGER_FLAG=LOGGER_FLAG,
        SAVE_FLAG=SAVE_FLAG, MULTENV_FLAG=MULTENV_FLAG,
        RENDER_FLAG=RENDER_FLAG, EVAL_FLAG=EVAL_FLAG)
