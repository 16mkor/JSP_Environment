import multiprocessing
import multiprocessing.connection
import numpy as np
from JSP_Environments.envs.production_env import ProductionEnv

def worker_process(remote: multiprocessing.connection.Connection, config: dict) -> None:
    """Executes the threaded interface to the environment.

    Arguments:
        remote {multiprocessing.connection.Connection} -- Parent thread
        config {dict} -- Configuration of the training environment
    """
    # Spawn training environment
    try:
        env = ProductionEnv('NO_PARAMETERS', np.random.randint(low=0, high=10**5), 1_000, 500, 'GTrXL-PPO')
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment process
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError
        except Exception as e:
            print(data)
            raise WorkerException(e)


class Worker:
    """A worker that runs one environment on one process."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process

    def __init__(self, env_config: dict):
        """
        Arguments:
            env_config {dict} -- Configuration of the training environment
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_config))
        self.process.start()


import tblib.pickling_support

tblib.pickling_support.install()
import sys


class WorkerException(Exception):
    """Exception that is raised in the worker process and re-raised in the main process."""

    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()
        print(sys.exc_info())
        super(WorkerException, self).__init__(str(ee))

    def re_raise(self):
        raise (self.ee, None, self.tb)
