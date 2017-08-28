import random
import numpy as np
import math
import logging
from datetime import datetime
import scipy.stats as stats

logger = logging.getLogger()


class SoftmaxPolicy(object):
    def __init__(self, dimension, num_actions):
        self.dimension = dimension
        self.num_actions = num_actions
        self.initialise_parameters()
        self.tiny = 1e-8

    def get_policy_parameters(self):
        return np.copy(self.parameters)

    def set_policy_parameters(self, parameters):
        self.parameters = parameters

    def initialise_parameters(self):
        """
        TODO: See different ways of initialising the parameters.
         - Zero vectors
         - Random vectors (capped to [-10, 10] for example)
         - Maximising log likelihood etc
        :return:
        """
        # self.parameters = np.random.uniform(low=self.tiny, high=1, size=(self.num_actions, self.dimension))
        self.parameters = np.zeros(shape=(self.num_actions * self.dimension), dtype=float)
        # self.parameters.fill(self.tiny)

    def get_num_actions(self):
        return self.num_actions

    def get_action(self, state_feature):
        '''
        Perform dot product between state feature and policy parameter and return sample from the normal distribution
        :param state_feature:
        :return:
        '''

        # for each policy parameter (representing each action)
        # calculate phi /cdot theta
        # put these into array and softmax and compute random sample
        action_probabilities = []
        policy_parameters = np.split(self.parameters, self.num_actions)
        for i, parameter in enumerate(policy_parameters):
            mu = np.dot(state_feature, parameter)
            action_probabilities.append(mu)

        # substract the largest value of actions to avoid erroring out when trying to find exp(value)
        max_value = action_probabilities[np.argmax(action_probabilities)]
        for i in range(len(action_probabilities)):
            action_probabilities[i] = action_probabilities[i] - max_value

        softmax = np.exp(action_probabilities) / np.sum(np.exp(action_probabilities), axis=0)

        running_total = 0.0
        total = np.zeros(shape=self.num_actions)
        for i, value in enumerate(softmax):
            running_total += value
            total[i] = running_total

        rand = random.uniform(0, 1)
        chosen_policy_index = 0
        for i in range(len(total)):
            if total[i] > rand:
                chosen_policy_index = i
                break

        return chosen_policy_index, softmax

    def dlogpi(self, state_feature, action):
        """
        Add delta to policy parameters. one component at a time.
        Then calculate the probability of producing the action

        :param state_feature:
        :param action:
        :return:
        """
        _, pi = self.get_action(state_feature)

        dlogpi_parameters = np.empty(self.num_actions, dtype=object)
        # for the theta parameter used for action (use index)
        for i in range(self.num_actions):
            if i == action:
                dlogpi_parameters[i] = np.dot((1 - pi[action]), state_feature)
            else:
                theta_x = self.parameters[self.dimension * i: self.dimension * (i + 1)]
                theta_action = self.parameters[self.dimension * action: self.dimension * (action + 1)]
                component1 = -1.0 * pi[action] * (np.exp(np.dot(theta_x, state_feature))/np.exp(np.dot(theta_action, state_feature)))
                dlogpi_parameters[i] = np.dot(component1, state_feature)

        return np.concatenate(dlogpi_parameters)

    def update_parameters(self, delta):
        self.set_policy_parameters(self.parameters + delta)
