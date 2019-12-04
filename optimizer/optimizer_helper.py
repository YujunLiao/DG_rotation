from torch import optim


def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    '''

    :param network:
    :param epochs:
    :param lr:
    :param train_all:
    :param nesterov:
    :return:
    '''

    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    #optimizer = optim.Adam(params, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler
