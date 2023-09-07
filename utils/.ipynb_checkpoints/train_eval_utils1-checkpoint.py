import sys
import torch
import math
import numpy as np
import random,os
from tqdm import tqdm

from utils.distributed_utils import reduce_value, is_main_process



def torch_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def adjust_learning_rate(optimizer, warm_up, epoch, epochs, base_lr, step, iteration_per_epoch):
    
    '''
    warm_up: number of steps
    '''
    
    T = epoch * iteration_per_epoch + step
    
    if T < warm_up:
        lr = base_lr  * T / warm_up
    else:
        min_lr = 0
        T = epoch * iteration_per_epoch - warm_up
        total_iters = epochs * iteration_per_epoch - warm_up
        lr = 0.5 * (1 + math.cos(1.0 * T / total_iters * math.pi)) * (base_lr - min_lr) + min_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    
def train_one_epoch(model, optimizer, data_loader,loss_function, device, epoch, epochs, warm_up, base_lr):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)
        
    for step, (images, labels) in enumerate(data_loader):
        
        if is_main_process():
            adjust_learning_rate(optimizer, warm_up, epoch, epochs, base_lr, step, len(data_loader))
        
        images, labels = images.to(device), labels.to(device)

        pred = model(images)
        loss = loss_function(pred, labels)
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()
        
        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))
        
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    
    sum_num = reduce_value(sum_num, average=False)
    
    return mean_loss.item(), sum_num.item()


@torch.no_grad()
def evaluate(model, data_loader,loss_function, device):
    model.eval()

    # 用于存储预测正确的样本个数
    mean_loss = torch.zeros(1).to(device)
    sum_num = torch.zeros(1).to(device)
    
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        pred = model(images)
        loss = loss_function(pred, labels)
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels).sum()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return mean_loss.item(), sum_num.item()






