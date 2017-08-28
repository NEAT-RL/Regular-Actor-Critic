from ValueFunction.ValueFunction import ValueFunction
from Policy.SoftmaxPolicy import SoftmaxPolicy
import numpy as np
from . import Feature
import math
"""
Mountain car
state = (position, velocity)
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        min speed = -self.max_speed = -0.07
         
TODO: 
Add Cartpole settings
"""


class RACAgent(object):
    def __init__(self, dimension, num_actions, state_length, feature_size):
        self.dimension = dimension
        # self.feature = self.__create_discretised_feature(state_length, feature_size)
        self.feature = self.__create_discretised_feature([10, 10, 10, 10], 4, [-2.4, -10, -41.8 * math.pi / 180.0, -10],
                                                         [2.4, 10, 41.8 * math.pi / 180.0, 10])
        self.valueFunction = ValueFunction(dimension)
        self.policy = SoftmaxPolicy(dimension, num_actions)
        self.fitness = 0
        self.beta = 0.005

    @staticmethod
    def __create_discretised_feature(partition_size, state_length, state_lower_bounds, state_upper_bounds):
        """
        :param partition_size: Array of partition_sizes for each state field
        :param state_length: Number of states == input dimension (state)
        :param state_lower_bounds: array of lower bounds for each state field
        :param state_upper_bounds: array of upper bounds for each state field
        :return: discretised feature
        """
        intervals = []
        output_dimension = 0

        for i in range(state_length):
            output_dimension += partition_size[i]
            state_col = Feature.DiscretizedFeature.create_partition(state_lower_bounds[i],
                                                                    state_upper_bounds[i], partition_size[i])
            intervals.append(state_col)

        # return Feature.DiscretizedFeature(state_length, output_dimension)
        return Feature.DiscretizedFeature(state_length, output_dimension, intervals)

    def get_feature(self):
        return self.feature

    def get_value_function(self):
        return self.valueFunction

    def get_policy(self):
        return self.policy

    def get_fitness(self):
        return self.fitness

    def update_parameters(self, state, action, reward, next_state):
        phi = self.feature.phi(state)
        phi_next = self.feature.phi(next_state)
        # calculate td_error
        td_error = self.valueFunction.td_error(reward, phi, phi_next)
        self.valueFunction.update_parameters(phi, td_error)

        self.policy.update_parameters(np.dot(self.beta * td_error, self.policy.dlogpi(phi, action)))
