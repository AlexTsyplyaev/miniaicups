import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def _torch_var(value):
    """Convert to pytorch Variable"""
    return Variable(torch.FloatTensor(value))


def _to_list(value):
    """Convert to list"""
    if isinstance(value, list):
        return value
    return [value]


class Agent(object):
    """Q-learning agent"""

    def __init__(self, network, solver='sgd', loss='mse', lr=0.01):
        """Init method"""
        self._net = network
        solvers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        losses = {
            'mse': nn.MSELoss,
            'cross-entropy': nn.CrossEntropyLoss
        }
        self._solver = solvers[solver](self._net.parameters(), lr=lr)
        self._epsilon = 0.3
        self._loss = losses[loss]()

    def fit(self, X, y):
        """Train agent"""
        X = _to_list(X)
        y = _to_list(y)
        self._net.train()
        for data, target in zip(X, y):
            data = _torch_var(data)
            target = _torch_var(target)
            # train:
            self._solver.zero_grad()
            predicted = self._net(data)
            loss = self._loss(predicted, target)
            loss.backward()
            self._solver.step()

    def predict(self, X):
        """Make action"""
        # TODO: epsilon greedy policy
        self._net.eval()
        X = _torch_var(X)
        q_values = self._net(X).data.numpy()
        return q_values
