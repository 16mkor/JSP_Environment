from JSP_env.envs.time_calc import *
from JSP_env.envs.heuristics import *
from JSP_env.envs.resources import *
from JSP_env.envs.transport import *
from JSP_env.envs.order import *
import simpy


class Sink(Resource):
    buffer_in = []

    def __init__(self, env, id, statistics, parameters, resources, agents, time_calc, location, label):
        Resource.__init__(self, statistics, parameters, resources, agents, time_calc, location)
        print("Sink %s created" % id)
        self.env = env
        self.id = id
        self.label = label
        self.type = "sink"
        self.buffer_in_indiv = []

    def put_buffer_in(self, order):
        self.buffer_in_indiv.append(order)
        Sink.buffer_in.append(order)
        order.order_log.append(["sink", order.id, round(self.env.now, 5), self.id])
        if len(Sink.buffer_in) >= self.parameters['NUM_ORDERS'] - 1:
            print("All orders processed")
            self.parameters['stop_criteria'].succeed()

    def is_free(self):
        return True

    def is_free_machine_group(self):
        return True