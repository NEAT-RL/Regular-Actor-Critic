import numpy as np


class ValueFunction(object):
    def __init__(self, dimension):
        self.dimension = dimension
        self.parameters = np.zeros(dimension, dtype=float)
        self.alpha = 0.1
        self.gamma = 0.99

    def get_value(self, state_feature):
        return np.dot(self.parameters, state_feature)

    def get_parameter(self):
        return np.copy(self.parameters)

    def td_error(self, reward, current_state_feature, next_state_feature):
        return reward + self.gamma * np.dot(self.parameters, next_state_feature) - np.dot(self.parameters, current_state_feature)

    def update_parameters(self, current_state_feature, td_error):
        delta = np.dot(self.alpha * td_error, current_state_feature)
        new_parameters = self.parameters + delta
        self.parameters = new_parameters
