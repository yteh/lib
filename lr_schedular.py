import numpy as np

def timeBasedDecay(epoch, lr, decay_rate=1):
    """
    - epoch      : int, current epoch value
    - lr         : int or float, current learning rate
    - decay_rate : int or float, the decay rate for each epoch

    return float, decayed lr
    """

    return lr / (1 + decay_rate * epoch)


def stepBasedDecay(epoch, lr_0, decay_rate=0.5, r=100):
    """
    - epoch      : int, current epoch value
    - lr_0       : int or float, inital/default learning rate (unchanged)
    - decay_rate : int or float, the decay rate for each epoch

    return float, decayed lr
    """

    return lr_0 * (decay_rate ** ((1+epoch)/r))


def exponentialDecay(epoch, lr, decay_rate):
    """
    - epoch      : int, current epoch value
    - lr         : int or float, current learning rate
    - decay_rate : int or float, the decay rate for each epoch

    return float, decayed lr
    """

    return lr * np.exp(-decay_rate * epoch)