"""
PyTorch networks
"""
import torch.nn as nn


def _test_network(in_features, out_features):
    net = nn.Sequential()
    net.add_module('fc1', nn.Linear(in_features, 27))
    net.add_module('relu1', nn.ReLU())
    net.add_module('fc2', nn.Linear(27, 17))
    net.add_module('relu2', nn.ReLU())
    net.add_module('fc3', nn.Linear(17, 10))
    net.add_module('relu3', nn.ReLU())
    net.add_module('fc4', nn.Linear(10, out_features))
    return net


class Networks(object):
    """Main class for available networks"""
    def __init__(self):
        """Init method"""
        self._nets = {
            'test': _test_network,
        }

    def available(self):
        """Return available networks names"""
        return self._nets.keys()

    def __getitem__(self, name):
        """Return network by name"""
        return self._nets[name]
