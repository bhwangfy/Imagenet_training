import os
import math
import tempfile
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
from torchvision import transforms


import models
import optimizers
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from utils.train_eval_utils import train_one_epoch, evaluate, torch_seed




model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__") 
                                       and callable(models.__dict__[name]))

optimizer_names = sorted(name for name in optimizers.__dict__
                     if name.islower() and not name.startswith("__") 
                                  and callable(optimizers.__dict__[name]))

# add arguement
## model/optimizer/dataset
parser = argparse.ArgumentParser(description='Image Classification')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                                help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--optimizer', '-op', metavar='OPTIM', default='lion',choices=optimizer_names,
                                help='optimizer: ' + ' | '.join(optimizer_names) + ' (default: sgd)')
parser.add_argument('--dataset', '-data', metavar='DATASET', default='imagenet', 
                                help='dataset: cifar10/cifar100/imagenet (default: cifar10)')
parser.add_argument('--dataset-dir', default='/code/multi_gpu/dataset', type=str,
                                help='The directory for dataset')
parser.add_argument('--test', dest='test', action='store_true',default = False,
                                help='use train-validation-test, otherwise train-validation')
parser.add_argument('--syncBN', type=bool, default=False,
                                help = 'sync batch normalization, useful when batch size small')

## distributed training: single machine multiple gpus
parser.add_argument('--device', default='cuda', 
                                help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--world-size', default=4, type=int,
                                help='number of distributed processes, will be auto-set by nproc_per_node')
parser.add_argument('--dist-url', default='env://', 
                                help='url used to set up distributed training, \
                                      check https://pytorch.org/docs/stable/distributed.html for more information')

## hyperparameter
parser.add_argument('--seed', '-sd', default=1, type=int, 
                                help='random seed (default: 1)')
parser.add_argument('--epochs', default=100, type=int, 
                                help='number of total epochs to run (default: 100')
parser.add_argument('--start-epoch', default=0, type=int, 
                                help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=1024, type=int, metavar='N', 
                                help='mini-batch size (default: 1024)')
parser.add_argument('-learning-rate', '--lr', default=None, type=float, 
                                help='initial learning rate (default: None)')
parser.add_argument('--weight-decay', '-wd', default=0, type=float,
                                help='weight decay (default: 0)')
parser.add_argument('--hyperparameter', '-hp', default= None,type=float,nargs='*',
                                help='hyperparameter, None = use default setting in optimizer.py (default: None)')

## save model
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',default='/model/temp', type=str,
                                help='The directory used to save the trained models, 调试阶段 /code, 测试阶段 /model)')
parser.add_argument('--save-every', dest='save_every', type=int, default=10,
                                help='Saves checkpoints at every specified number of epochs')


best_acc = 0
best_epoch = 0

def main():
    global args, best_acc
    args = parser.parse_args()   
#    torch_seed(args.seed)
    
    # check devie, only process if gpu available
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
        
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # initialize distributed envirionment
    init_distributed_mode(args=args)

    rank = args.rank                  
    device = torch.device(args.device)     
    checkpoint_path = ""              

    if rank == 0: 
        print(args,'\n')

    #load data
    
    print(f' => load dataset: [{args.dataset}] \n') if rank == 0 else print('\t')
    
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                     (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,])
        
    val_transform = transforms.Compose([transforms.ToTensor(),normalize,])
        
    '''
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])
    '''

    if args.dataset == 'cifar10':
        num_classes = 10
        train_data_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, 
                                                      train=True,
                                                      download=True, 
                                                      transform=train_transform)

        val_data_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, 
                                                    train=False, 
                                                    download=True, 
                                                    transform=val_transform)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_data_set = torchvision.datasets.CIFAR100(root=args.dataset_dir, 
                                                       train=True,
                                                       download=True, 
                                                       transform=train_transform)

        val_data_set = torchvision.datasets.CIFAR100(root=args.dataset_dir, 
                                                     train=False, 
                                                     download=True, 
                                                     transform=val_transform)
    
    elif args.dataset == 'imagenet':
        args.print_freq = True
        num_classes = 1000
        print('get train')  if rank == 0 else print('\t')
        train_data_set = torchvision.datasets.ImageFolder(root='/dataset/ImageNet2012/train',
                                                         transform=train_transform)
        print('get validation')  if rank == 0 else print('\t')
        val_data_set = torchvision.datasets.ImageFolder(root='/dataset/ImageNet2012/val',
                                                       transform=val_transform)


    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    
    # number of workers
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  
    
    print(' => Using {} dataloader workers every process'.format(nw)) if rank == 0 else print('\t')
        
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)
    # build model
    print(f' => build model: [{args.arch}]') if rank == 0 else print('\t')
    model = models.__dict__[args.arch](num_classes = num_classes ).to(device)
    

    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))  if rank == 0 else print('\t')
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint(epoch {})"
                  .format(checkpoint['epoch']))  if rank == 0 else print('\t')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))  if rank == 0 else print('\t')
    else:
        # for 1st process, initialize and save model weights
        # for other process, load weights of 1st process to use the same initialization for all process
        # map_location to keep balanced usage of gpu
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        if rank == 0:      
            torch.save(model.state_dict(), checkpoint_path) 
        dist.barrier()  
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define optimizer
#    pg = [p for p in model.parameters() if p.requires_grad]
    
    if args.optimizer == 'sgd':
        if args.lr == None:
            args.lr = 1e-2
        if args.hyperparameter == None:
            args.hyperparameter = [0.9]
        args.lr *= args.world_size  
        optimizer = optimizers.sgd(model.parameters(), 
                                    args.lr,
                                    momentum=args.hyperparameter[0],
                                    weight_decay=args.weight_decay)

    elif args.optimizer == 'adam':
        if args.lr == None:
            args.lr = 1e-3
        if args.hyperparameter == None:
            args.hyperparameter = [0.9,0.999,1e-8]
        args.lr *= args.world_size  
        optimizer = optimizers.adam(model.parameters(), 
                                     args.lr, 
                                     betas=(args.hyperparameter[0],
                                            args.hyperparameter[1]),
                                     eps=args.hyperparameter[2],
                                     weight_decay=args.weight_decay)
        
    elif args.optimizer == 'lion':
        if args.lr == None:
            args.lr = 1e-4
        if args.hyperparameter == None:
            args.hyperparameter = [0.9,0.99]
        args.lr *= args.world_size  
        optimizer = optimizers.lion(model.parameters(), 
                                      args.lr,
                                      betas=(args.hyperparameter[0],
                                             args.hyperparameter[1]), 
                                      weight_decay=args.weight_decay)
        
    elif args.optimizer == 'signsgd':
        if args.lr == None:
            args.lr = 1e-4
        if args.hyperparameter == None:
            args.hyperparameter = [0.9]
        args.lr *= args.world_size  
        optimizer = optimizers.signsgd(model.parameters(), 
                                      args.lr,
                                      betas=(args.hyperparameter[0]),
                                      weight_decay=args.weight_decay)
    
    print(f' => define optimizer: \n [{optimizer}] \n')  if rank == 0 else print('\t')
    
    scheduler = lr_scheduler.MultiStepLR(optimizer,milestones = [60,60])
    
    criterion = torch.nn.CrossEntropyLoss()
    #.to(device)
    
    print('\t')
    print('=> Start training..')  if rank == 0 else print('\t')
    
    Train_acc = []
    Train_loss = []
    Val_acc = []
    Val_loss = []
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)

        tloss, tsum_num = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    loss_function= criterion,
                                    device=device,
                                    epoch=epoch)
        tacc = tsum_num / train_sampler.total_size * 100
#        print('==>  [epoch {}] Train  \t '
#              ' Loss [{:.4f}] \t '
#              'Accuracy [{:.2f}]\t'.format(epoch, tloss, tacc))  if rank == 0 else print('\t')
        scheduler.step()

        vloss, vsum_num = evaluate(model=model,
                           data_loader=val_loader,
                           loss_function= criterion,
                           device=device)
        vacc = vsum_num / val_sampler.total_size * 100
        
        scheduler.step()
        
        Train_acc.append(tacc)
        Train_loss.append(tloss)
        Val_acc.append(vacc)
        Val_loss.append(vloss)
        
        if rank == 0:
            print('[epoch {}] Validation  \t Loss [{:.4f}] \t Accuracy [{:.2f}]\t'.format(epoch, 
                                                                                                   vloss, 
                                                                                                   vacc))
            if vacc > best_acc:
                print('======> Validation accuracy improved by {:.6f}'.format(vacc-best_acc))
                best_acc = vacc
                best_epoch = epoch

            state = {'state_dict': model.module.state_dict(),
                     'epoch': epoch,
                     'best_acc': best_acc,
                     'best_epoch': best_epoch,}
            torch.save(state, os.path.join(args.save_dir,f'model-{epoch}.pth'))
            np.savetxt(os.path.join(args.save_dir,'train_acc.txt'), Train_acc)
            np.savetxt(os.path.join(args.save_dir,'train_loss.txt'), Train_loss)
            np.savetxt(os.path.join(args.save_dir,'val_acc.txt'), Val_acc)
            np.savetxt(os.path.join(args.save_dir,'val_loss.txt'), Val_loss)

    # delete tempfile
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    main()
