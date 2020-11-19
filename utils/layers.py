from torch import nn
from torch.nn import init


def init_weights(model):
    if type(model) in [nn.Linear, nn.Conv2d]:
        init.xavier_uniform_(model.weight)
        init.constant_(model.bias, 0)
    elif type(model) in [nn.LSTMCell]:
        init.constant_(model.bias_ih, 0)
        init.constant_(model.bias_hh, 0)
