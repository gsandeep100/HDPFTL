from models.BayesianTabularNet import BayesianTabularNet
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from torch.optim import Adam


def extract_priors(model):
    return {
        'fc1.weight': model.fc1.weight.detach(),
        'fc1.bias': model.fc1.bias.detach(),
        'fc2.weight': model.fc2.weight.detach(),
        'fc2.bias': model.fc2.bias.detach(),
    }


"""
def train_bayesian_local(X_train, y_train, input_dim, num_classes, prior_params, device='cpu'):
    model = BayesianTabularNet(input_dim, num_classes, prior_params=prior_params).to(device)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    for step in range(200):
        svi.step(X_train, y_train)

    return guide
"""


def train_bayesian_local(X_train, y_train, input_dim, num_classes, prior_params, device='cpu'):
    model = BayesianTabularNet(input_dim, num_classes, prior_params=prior_params).to(device)
    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, Adam({"lr": 1e-3}), loss=Trace_ELBO())

    for step in range(200):
        svi.step(X_train, y_train)

    return guide


def aggregate_guides(guides):
    avg_loc = {}
    count = len(guides)

    for name in guides[0].locs:
        avg_loc[name] = sum([g.locs[name] for g in guides]) / count

    return avg_loc
