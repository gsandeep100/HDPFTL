import pyro
import pyro.distributions as dist
import torch
from pyro.nn import PyroSample, PyroModule
from torch import nn


class BayesianTabularNet(PyroModule):
    def __init__(self, input_dim, num_classes, prior_std=0.1):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](input_dim, 128)
        self.fc1.weight = PyroSample(dist.Normal(0., prior_std).expand([128, input_dim]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., prior_std).expand([128]).to_event(1))

        self.fc2 = PyroModule[nn.Linear](128, num_classes)
        self.fc2.weight = PyroSample(dist.Normal(0., prior_std).expand([num_classes, 128]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., prior_std).expand([num_classes]).to_event(1))

    def forward(self, x, y=None):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits
