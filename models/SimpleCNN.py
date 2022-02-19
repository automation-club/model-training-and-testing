from torch import nn

import torch


class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes