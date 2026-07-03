import torch.nn as nn


def get_loss_function():

    """
    Returns the loss function.
    """

    criterion = nn.CrossEntropyLoss()

    return criterion