import torch
import torch.nn as nn
import time
import os
import math

from opts import opts
from utils import compute_par


class AverageMeter():
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count += n
        self.avg = self.sum / self.count
        

batch_time_train = AverageMeter()
data_time_train = AverageMeter()
pars_train = AverageMeter()
losses_d_train = AverageMeter()
losses_total_train = AverageMeter()

batch_time_test = AverageMeter()
pars_test = AverageMeter()


def train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model_source, model_target, criterion_d, optimizer, epoch, args):
    """
    Train for one epoch. Only a batch is used in a epoch, not all the batches.
    Parameters
    ----------
    train_loader_source: torch.utils.data.DataLoader
        Used to reset train_loader_source_batches if the enumerate reach the end of iteration
    train_loader_source_batches: enumerate 
        An object whose each element contain one batch of source data
    train_loader_target: torch.utils.data.DataLoader
        Used to reset train_loader_target_batches if the enumerate reach the end of iteration
    train_loader_target_batches: enumerate
        An object whose each element contain one batch of target data
    model_source: pytorch model
        The model in source training pipeline
    model_target: pytorch model
        The model in target training pipeline
    criterion_d: A certain class of loss in torch.nn
        The criterion of the domain classifier model        
    optimizer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model 
    args: Namespace
        Arguments that main.py receive
    epoch: int
        The current epoch
    Return
    ------
    par_train: float
        The pixel agreement ratio in this minibatch
    loss_total_train: float
        The loss in this minibatch
    """

    model.train()
    adjust_learning_rate(optimizer, epoch, args)
    end = time.time()
    # because the batch size of the last batch of a dataset is usually less than args.batch_size, so the real_labels and fake_labels can not simply set their size equals to args.batch_size

    # prepare the data for the model forward and backward
    try:
        _, (inputs_source, labels_source) = train_loader_source_batches.__next__()
    except StopIteration:
        train_loader_source_batches = enumerate(train_loader_source)
        _, (inputs_source, labels_source) = train_loader_source_batches.__next__()
    if torch.cuda.is_available():
        inputs_source = inputs_source.cuda(async=True)
        labels_source = labels_source.cuda(async=True)
    inputs_source_var, labels_source_var = torch.autograd.Variable(inputs_source), torch.autograd.Variable(labels_source)

    try: 
        _, (inputs_target, _) = train_loader_target_batches.__next__()
    except StopIteration:
        train_loader_target_batches = enumerate(train_loader_target)
        _, (inputs_target, _) = train_loader_target_batches.__next__()
    if torch.cuda.is_available():
        inputs_target = inputs_target.cuda(async=True)
    inputs_target_var = torch.autograd.Variable(inputs_target)
    data_time_train.update(time.time() - end)

    # calculate for the source and target data
    domain_outputs_source, domain_outputs_target = model(inputs_source_var)
    batch_size_source = inputs_source.size(0) # here batch_size_source is equal to batch_size_target
    fake_labels = torch.LongTensor(batch_size_source).fill_(0)
    real_labels = torch.LongTensor(batch_size_target).fill_(1)
    if torch.cuda.is_available():
        fake_labels = fake_labels.cuda()
        real_labels = real_labels.cuda()
    fake_labels = torch.autograd.Variable(fake_labels)  
    real_labels = torch.autograd.Variable(real_labels)         
    loss_d_source = criterion_d(domain_outputs_source, fake_labels) # 0 means from source domain and 1 means from target domain 
    loss_d_target = criterion_d(domain_outputs_target, real_labels)    

    # measure pixel agreement ratio and record loss
    par_train = compute_par(inputs_source, inputs_target)
    pars_train.update(par_train, batch_size_source)

    # calculate for the target data
    # measure accuracy and record loss    
    loss_d_train = loss_d_source + loss_d_target
    losses_d_train.update(loss_d_train.data, batch_size_source)
    loss_total_train = loss_d_train
    losses_total_train.update(loss_total_train, batch_size_source)
    model.zero_grad()
    loss_total_train.backward()
    # 这里要把权重共享的部分的梯度减半
    pass
    optimizer.step()
    batch_time_train.update(time.time() - end)

    if epoch % args.print_freq == 0:
        print('Tr epoch [{0}/{1}]\t'
              'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Par {pars.val:.3f} ({pars.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Loss_d {loss_d_train.val:.4f} ({loss_d_train.avg:.4f})\t'.format(
              epoch, args.epochs, batch_time=batch_time_train, data_time=data_time_train, pars=pars_train, loss=losses_total_train, 
              loss_d_train=losses_d_train)
              )
    if epoch % args.record_freq == 0:
        if not os.path.isdir(args.log):
            os.mkdir(args.log)
        with open(os.path.join(args.log, 'log.txt'), 'a+') as fp:
            fp.write('\n')
            fp.write('Tr:epoch: %d, par: %4f, loss_total: %4f,'
                     'loss_d: %4f'
                     % (epoch, pars_train.avg, losses_total_train.avg, losses_d_train.avg))    

    return par_train, loss_total_train


def test(model_source, model_target, criterion_d, epoch, args):
    """
    Test on the whole validation set
    Parameters
    ----------
    model_source: pytorch model
        The model in source training pipeline
    model_target: pytorch model
        The model in target training pipeline
    criterion_d: A certain class of loss in torch.nn
        The criterion of the domain classifier model  
    args: Namespace
        Arguments that main.py receive
    epoch: int
        The current epoch
    Return
    ------
    pars_val_tmp.avg: float
        The average pixel agreement ratio in validation set
    """
    model.eval()
    end = time.time()
    pars_testf_tmp = AverageMeter()
    # because the batch size of the last batch of a dataset is usually less than args.batch_size, so the real_labels and fake_labels can not simply set their size equals to args.batch_size
    with torch.no_grad():    
        for i in range(args.batch_size):
            gen_outputs_source, gen_outputs_target = model()
            par_test = compute_par(inputs_source, inputs_target)
            pars_test_tmp.update(par_test)

        batch_time_test.update(time.time() - end)
        end = time.time()
    pars_test.update(pars_test_tmp.avg)
    if epoch % args.print_freq == 0:
        print('Te epoch [{0}/{1}]\t'
            'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Par {pars.val:.3f} ({pars.avg:.3f})\t'
            .format(
            epoch, args.epochs, batch_time=batch_time_test, 
            pars=pars_test) 
            )

    print(' * Target Dataset par {pars.avg:.3f}'
        .format(pars=pars_test_tmp))
    if not os.path.isdir(args.log):
        os.mkdir(args.log)
    with open(os.path.join(args.log, 'log.txt'), 'a+') as fp:
        fp.write('\n')
        fp.write('    Test Target: epoch %d, par: %4f'
                    % (epoch, pars_test.avg))

    return pars_test_tmp.avg


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjust the learning rate according to the epoch
    Parameters
    ----------
    optimzer: An optimizer in a certain update principle in torch.optim
        The optimizer of the model 
    epoch: int
        The current epoch
    args: Namespace
        Arguments that main.py receive
    Return
    ------
    The function has no return
    """    
    exp = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 1 or 0
    exp_pretrain = epoch > args.schedule[4] and 5 or epoch > args.schedule[3] and 4 or epoch > args.schedule[2] and 3 or epoch > args.schedule[1] and 2 or epoch > args.schedule[0] and 2 or 2
    lr = args.lr * (args.gamma ** exp)
    lr_pretrain = args.lr * (args.gamma ** exp_pretrain)
    for param_group in optimizer.param_groups:
        if param_group['name'] == 'pre-trained':
            # param_group['lr'] = exp_pretrain
            param_group['lr'] = 1e-3
        else:
            param_group['lr'] = lr