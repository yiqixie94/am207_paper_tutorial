import os
import torch


def adjust_learning_rate(optimizer, lr):
    '''adjust the learning rate of the `optimizer` to `lr`'''
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(dir, epoch, **kwargs):
    '''package the `kwargs` with the epoch number and apply torch.save'''
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


def train_epoch(loader, model, criterion, optimizer):
    '''train for a single epoch
            loader:     torch loader of the training set
            model:      the torch model to train
            criterion:  the criterion of loss
            optimizer:  the torch optimizer to do optimization
    '''

    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        # preprocess
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # evaluate
        output = model(input_var)
        loss = criterion(output, target_var)
        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # record
        loss_sum += loss.data[0] * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def eval(loader, model, criterion):
    '''evaluate the `model` on the dataset given by `loader` with `criterion`'''
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    for i, (input, target) in enumerate(loader):
        # preprocess
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # evaluate
        output = model(input_var)
        loss = criterion(output, target_var)
        # record
        loss_sum += loss.data[0] * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }


def moving_average(net1, net2, alpha=1):
    '''aggregate the parameters of `net2` onto `net1`:
            net1:   the swa model, aggregator
            net2:   the sgd model
            alpha:  the factor indicating the proportion of training not done'''
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def _check_bn(module, flag):
    '''check if `module` has batch normalize buffers
        result will be recorded in the `flag`'''
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):
    '''check if `model` has batch normalize buffers
        by applying _check_bn() on all of its modules'''
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    '''reset batch normalize buffers following:
            running means are reset to 0s
            running variances are reset to 1s'''
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    '''add the momenta info (if any) of `module` to the `momenta` dictionary'''
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    '''set the momenta info (if any) of `module` from the `momenta` dictionary'''
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    for input, _ in loader:
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        b = input_var.data.size(0)

        momentum = b / (n + b)
        for module in momenta.keys():
            module.momentum = momentum

        model(input_var)
        n += b

    model.apply(lambda module: _set_momenta(module, momenta))
