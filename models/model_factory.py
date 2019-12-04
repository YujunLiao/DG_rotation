from models import caffenet
from models import mnist
from models import patch_based
from models import alexnet
from models import resnet

nets_map = {
    'caffenet': caffenet.caffenet,
    'alexnet': alexnet.alexnet,
    'resnet18': resnet.resnet18,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}


def get_network(name):
    """Return the function of getting the network.

    :param name: Name of the network used for extract features.
    :return:{function}
    """
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        """

        :param kwargs:
        :return:
        """
        return nets_map[name](**kwargs)

    return get_network_fn
