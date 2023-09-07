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

# arguement
## model/optimizer/dataset
parser = argparse.ArgumentParser(description='Image Classification')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                                help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--optimizer', '-op', metavar='OPTIM', default='lion',choices=optimizer_names,
                                help='optimizer: ' + ' | '.join(optimizer_names) + ' (default: lion)')
parser.add_argument('--dataset', '-data', metavar='DATASET', default='imagenet', 
                                help='dataset: cifar10/cifar100/imagenet (default: cifar10)')
parser.add_argument('--dataset-dir', default='/code/dataset', type=str,
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
parser.add_argument('--epochs', default=90, type=int, 
                                help='number of total epochs to run (default: 90')
parser.add_argument('--start-epoch', default=0, type=int, 
                                help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', '-bs', default=1024, type=int, metavar='N', 
                                help='mini-batch size (default: 1024)')
parser.add_argument('--learning-rate', '-lr', default=None, type=float, 
                                help='initial learning rate (default: None)')
parser.add_argument('--warmup', type=int, default=10000)
parser.add_argument('--weight-decay', '-wd', default=0, type=float,
                                help='weight decay (default: 0)')
parser.add_argument('--hyperparameter', '-hp', default= None,type=float,nargs='*',
                                help='hyperparameter, None = use default setting in optimizer.py (default: None)')

## save model
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', dest='save_dir',default='/code/distributed/temp/', type=str,
                                help='The directory used to save the trained models, 调试阶段 /code, 测试阶段 /model)')
parser.add_argument('--save-every', dest='save_every', type=int, default=10,
                                help='Saves checkpoints at every specified number of epochs')


best_acc = 0
best_epoch = 0

def main():
    global args, best_acc, best_epoch
    args = parser.parse_args()   
    torch_seed(args.seed)
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError("no GPU device for training.")

    # initialize distributed envirionment
    init_distributed_mode(args=args)

    rank = args.rank                  
    device = torch.device(args.device)     
    checkpoint_path = ""              

    if rank == 0: 
        print(args,'\n')
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    #load data 
    print(f' => load dataset: [{args.dataset}] \n') if rank == 0 else print('\t')
    
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
   

    if args.dataset == 'cifar10':
        nclass = 10
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,])
        
        val_transform = transforms.Compose([transforms.ToTensor(),normalize,])
    
        train_data_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, 
                                                      train=True,
                                                      download=True, 
                                                      transform=train_transform)
        
        val_data_set = torchvision.datasets.CIFAR10(root=args.dataset_dir, 
                                                    train=False, 
                                                    download=True, 
                                                    transform=val_transform)
    elif args.dataset == 'cifar100':
        nclass = 100
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomCrop(32, 4),
                                              transforms.ToTensor(),
                                              normalize,])
        val_transform = transforms.Compose([transforms.ToTensor(),normalize,])
        
        train_data_set = torchvision.datasets.CIFAR100(root=args.dataset_dir, 
                                                       train=True,
                                                       download=True, 
                                                       transform=train_transform)

        val_data_set = torchvision.datasets.CIFAR100(root=args.dataset_dir, 
                                                     train=False, 
                                                     download=True, 
                                                     transform=val_transform)
    
    elif args.dataset == 'imagenet':
        nclass = 1000
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize,])

        val_transform = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            normalize])
        
        train_data_set = torchvision.datasets.ImageFolder(root='/blob/ImageNet/train',
                                                         transform=train_transform)

        val_data_set = torchvision.datasets.ImageFolder(root='/blob/ImageNet/val',
                                                       transform=val_transform)


    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # number of workers
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  
    
    print(' => use {} dataloader workers every process'.format(nw)) if rank == 0 else print('\t')
        
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               shuffle=(train_sampler is None),
                                               pin_memory=True,
                                               num_workers=nw)
    print(len(train_loader))

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=args.batch_size,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)
    # build model
    print(f' => build model: [{args.arch}]') if rank == 0 else print('\t')
    model = torchvision.models.__dict__[args.arch]().to(device)
    

    # resume model
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))  if rank == 0 else print('\t')
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint(epoch {})".format(checkpoint['epoch'])) if rank == 0 else print('\t')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume)) if rank == 0 else print('\t')
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
    
    if args.optimizer == 'sgd':
        if args.learning_rate == None:
            args.learning_rate = 1e-1
        if args.hyperparameter == None:
            args.hyperparameter = [0.9]
        # args.learning_rate *= args.world_size  
        optimizer = optimizers.sgd(model.parameters(), 
                                    args.learning_rate,
                                    momentum=args.hyperparameter[0],
                                    weight_decay=args.weight_decay)

    elif args.optimizer == 'adam':
        if args.learning_rate == None:
            args.learning_rate = 1e-3
        if args.hyperparameter == None:
            args.hyperparameter = [0.9,0.999,1e-8]
        # args.learning_rate *= args.world_size  
        optimizer = optimizers.adam(model.parameters(), 
                                     args.learning_rate, 
                                     betas=(args.hyperparameter[0],
                                            args.hyperparameter[1]),
                                     eps=args.hyperparameter[2],
                                     weight_decay=args.weight_decay)
        
    elif args.optimizer == 'lion':
        if args.learning_rate == None:
            args.learning_rate = 1e-4
        if args.hyperparameter == None:
            args.hyperparameter = [0.9,0.99]
        # args.learning_rate *= args.world_size  
        optimizer = optimizers.lion(model.parameters(), 
                                      args.learning_rate,
                                      betas=(args.hyperparameter[0],
                                             args.hyperparameter[1]), 
                                      weight_decay=args.weight_decay)
        
    elif args.optimizer == 'signsgd':
        if args.learning_rate == None:
            args.learning_rate = 1e-4
        if args.hyperparameter == None:
            args.hyperparameter = [0.99]
        # args.learning_rate *= args.world_size  
        optimizer = optimizers.signsgd(model.parameters(), 
                                      args.learning_rate,
                                      betas=(args.hyperparameter[0]),
                                      weight_decay=args.weight_decay)
    
    print(f' => define optimizer: \n [{optimizer}] \n')  if rank == 0 else print('\t')

    # resume optimizer ad scheduler
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))  if rank == 0 else print('\t')
            checkpoint = torch.load(args.resume)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint(epoch {})"
                  .format(checkpoint['epoch']))  if rank == 0 else print('\t')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))  if rank == 0 else print('\t')
            
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    print('\t')
    print('=> Start training..') if rank == 0 else print('\t')
    
    Train_acc = []
    Train_loss = []
    Val_acc = []
    Val_loss = []
    LR = []
    
    info = f'{args.dataset}_{args.optimizer}_sd_{args.seed}_n_{args.epochs}_bs_{args.batch_size}_lr_{args.learning_rate}_warmup_{args.warmup}_wd_{args.weight_decay}_hp_{args.hyperparameter}'
    
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        
        print(f' => [{optimizer}] \n')  if rank == 0 else print('\t')

        tloss, tsum_num, epoch_lr = train_one_epoch(model=model,
                                          optimizer=optimizer,
                                          data_loader=train_loader,
                                          loss_function= criterion,
                                          device=device,
                                          epoch=epoch,
                                          epochs = args.epochs,
                                          warm_up = args.warmup, 
                                          base_lr = args.learning_rate)
        tacc = tsum_num / train_sampler.total_size * 100
#        print('==>  [epoch {}] Train  \t '
#              ' Loss [{:.4f}] \t '
#              'Accuracy [{:.2f}]\t'.format(epoch, tloss, tacc))  if rank == 0 else print('\t')

        vloss, vsum_num = evaluate(model=model,
                           data_loader=val_loader,
                           loss_function= criterion,
                           device=device)
        vacc = vsum_num / val_sampler.total_size * 100
        
        
        Train_acc.append(tacc)
        Train_loss.append(tloss)
        Val_acc.append(vacc)
        Val_loss.append(vloss)
        LR.extend(epoch_lr)
        if rank == 0:
            print('[epoch {}] Train  \t Loss [{:.4f}] \t Accuracy [{:.2f}]\t'.format(epoch,tloss,tacc))
            print('[epoch {}] Validation  \t Loss [{:.4f}] \t Accuracy [{:.2f}]\t'.format(epoch,vloss,vacc))
            if vacc > best_acc:
                print('======> Validation accuracy improved by {:.6f}'.format(vacc-best_acc))
                best_acc = vacc
                best_epoch = epoch

                state = {'state_dict': model.module.state_dict(),
                         'optimizer':optimizer.state_dict(),
                         'epoch': epoch,
                         'best_acc': best_acc,
                         'best_epoch': best_epoch,}
                torch.save(state, os.path.join(args.save_dir,f'{info}_best.checkpoint.pth'))
            np.savetxt(os.path.join(args.save_dir,f'{info}_train.acc.txt'), Train_acc)
            np.savetxt(os.path.join(args.save_dir,f'{info}_train.loss.txt'), Train_loss)
            np.savetxt(os.path.join(args.save_dir,f'{info}_val.acc.txt'), Val_acc)
            np.savetxt(os.path.join(args.save_dir,f'{info}_val.loss.txt'), Val_loss)
            np.savetxt(os.path.join(args.save_dir,f'{info}_lr.txt'), LR)
    
    state = {'state_dict': model.module.state_dict(),
                         'optimizer':optimizer.state_dict(),
                         'epoch': epoch,
                         'best_acc': best_acc,
                         'best_epoch': best_epoch,}
    torch.save(state, os.path.join(args.save_dir,f'{info}_last.checkpoint.pth'))
    print(f"Finish training, best validation accuracy {best_acc} at epoch {best_epoch}")
    # delete tempfile
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()


if __name__ == '__main__':
    main()
