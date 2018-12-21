import os
import json
import shutil
import torch.optim
import torch.nn as nn
import random
import numpy as np
import torch.backends.cudnn as cudnn
import ipdb

from models.resnet import resnet  # The model construction
from trainer import train  # For the training process
from trainer import validate  # For the validate (test) process
from opts import opts  # The options for the project
from data.prepare_data import generate_dataloader  # Prepare the data and dataloader



def main():
    args = opts()
    # 将每一个epoch洗牌后的序列固定, 以使多次训练的过程中不发生较大的变化(到同一个epoch时会得到同样的模型)
    if args.seed != 666:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        else:
            torch.manual_seed(args.seed)
    else: 
        if torch.cuda.is_available():
            torch.cuda.manual_seed(666)
        else:
            torch.manual_seed(666)  
    model_source, model_target = CoGAN(args)
    # define-multi GPU
    model_source = torch.nn.DataParallel(model_source).cuda()
    model_target = torch.nn.DataParallel(model_target).cuda()
    print('the memory id should be same')
    print(id(model_source.module.resnet_conv))   # the memory is shared here
    print(id(model_target.module.resnet_conv))
    print('the memory id should be different')
    print(id(model_source.module.fc))  # the memory id shared here.
    print(id(model_target.module.fc))
    # define loss function(criterion) and optimizer
    if torch.cuda.is_available():
        criterion_d = nn.CrossEntropyLoss().cuda()
    else:
        criterion_d = nn.CrossEntropyLoss()
    best_par = 0
    # To apply different learning rate to different layer
    """ optimizer这里还没有修改"""
    if args.pretrained:
        print('the pretrained setting of optimizer')
        if args.auxiliary_dataset == 'imagenet':
            optimizer = torch.optim.SGD([
                {'params': model_source.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_source.module.fc.parameters(), 'name': 'pre-trained'},
                {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
            ],
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        elif args.auxiliary_dataset == 'l_bird':
            optimizer = torch.optim.SGD([
                {'params': model_source.module.resnet_conv.parameters(), 'name': 'pre-trained'},
                {'params': model_source.module.fc.parameters(), 'name': 'new-added'},
                {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
            ],
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    else:
        print('the from scratch setting of optimizer')
        optimizer = torch.optim.SGD([
            {'params': model_source.module.resnet_conv.parameters(), 'name': 'new-added'},
            {'params': model_source.module.fc.parameters(), 'name': 'new-added'},
            {'params': model_target.module.fc.parameters(), 'name': 'new-added'},
        ],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    # optimizer_target = torch.optim.SGD([
    #     {'params': model_target.module.resnet_conv.parameters(), 'name': 'pre-trained'},
    #     {'params': model_target.module.fc.parameters(), 'lr': args.lr*10, 'name': 'new-added'}
    # ],
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # raise ValueError('the resume function is not finished')
            print("==> loading checkpoints '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_par = checkpoint['best_par']
            model_source.load_state_dict(checkpoint['source_state_dict'])
            model_target.load_state_dict(checkpoint['target_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("==> loaded checkpoint '{}'(epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('The file to be resumed from is not exited', args.resume)
    else:
        if not os.path.isdir(args.log):
            os.makedirs(args.log)
        log = open(os.path.join(args.log, 'log.txt'), 'w')
        state = {k: v for k, v in args._get_kwargs()}
        log.write(json.dumps(state) + '\n')
        log.close()

    cudnn.benchmark = True
    # process the data and prepare the dataloaders.
    train_loader_source, train_loader_target = generate_dataloader(args)
    # test only
    if not args.train:
        par_test = test(model_source, model_target, criterion_d, epoch, args)
        return
    print('begin training')
    train_loader_source_batch = enumerate(train_loader_source)
    train_loader_target_batch = enumerate(train_loader_target)

    writer = SummaryWriter(log_dir=args.log)
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        par_train, loss = train(train_loader_source, train_loader_source_batches, train_loader_target, train_loader_target_batches, model_source, model_target, criterion_d, optimizer, epoch, args)
        writer.add_scalars('data/scalar_group', {'par_train': pred1_acc_train,
                                                'loss': loss}, epoch)        
        # evaluate on the test data
        if (epoch + 1) % args.test_freq == 0:
            par_test = test(model_source, model_target, criterion_d, epoch, args)
            writer.add_scalars('data/scalar_group', {'par_test': par_test}, epoch)               
            is_best = par_test > best_par
            if is_best:
                best_par = par_test
                with open(os.path.join(args.log, 'log.txt'), 'a') as fp:
                    fp.write('      \nTarget_T1 acc: %3f' % (best_par))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'model_state_dict': model.state_dict(),
                'best_par': best_par,
                'optimizer': optimizer.state_dict()
                },
                is_best, args, epoch + 1)
    writer.close()


def save_checkpoint(states, is_best, args, epoch):
    """
    Save the current model in the end of every epoch, and maintain the best model in training procedure
    Parameters
    ----------
    states: dict
        States needed to be saved
    is_best: bool
        Indicator of whether the current model is the best one, True for yes
    args: Namespace
        Arguments that main.py receive
    epoch: int
        The current epoch
    Returns
    -------
    This function has no return
    """
    filename = str(epoch) + '_checkpoint.pth.tar'
    dir_save_file = os.path.join(args.log, filename)
    torch.save(states, dir_save_file)
    if is_best:
        shutil.copyfile(dir_save_file, os.path.join(args.log, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()





