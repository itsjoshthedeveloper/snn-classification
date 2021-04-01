#############################################
#   @author: Josh Chough and Youngeun Kim   #
#############################################
#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from   torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

cudnn.enabled = False
cudnn.benchmark = True
cudnn.deterministic = True

import wandb

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import numpy as np
import datetime
import sys
import os

from utils import *
from test import test

# os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
p = argparse.ArgumentParser(description='Training ANN/SNN for classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Processing
p.add_argument('--seed',            default=0,                  type=int,       help='Random seed')
p.add_argument('--num_workers',     default=4,                  type=int,       help='number of workers')

# Wandb and file
p.add_argument('--wandb_mode',      default='online',           type=str,       help='wandb mode', choices=['online','offline','disabled'])
p.add_argument('--project',         default='snn',              type=str,       help='project name')
p.add_argument('--file_name',       default='',                 type=str,       help='Add-on for the file name')

# Model
p.add_argument('--model_type',      default='snn',              type=str,       help='model type', choices=['ann','snn'])
p.add_argument('--arch',            default='vgg16',            type=str,       help='architecture', choices=['vgg5','vgg9','vgg11','vgg16'])
p.add_argument('--kernel_size',     default=3,                  type=int,       help='filter size for the conv layers')
p.add_argument('--pretrained',      default=True, const=True,   type=str2bool,  help='use pretrained parameters from torchvision if possible', nargs='?')

# Dataset
p.add_argument('--dataset',         default='cifar10',          type=str,       help='dataset', choices=['cifar10','cifar100'])
p.add_argument('--batch_size',      default=16,                 type=int,       help='Batch size')
p.add_argument('--img_size',        default=64,                 type=int,       help='Image size')
p.add_argument('--augment',         action='store_true',                        help='turn on data augmentation')
p.add_argument('--attack',          default='',                 type=str,       help='adversarial attack', choices=['saltpepper','gaussiannoise'])
p.add_argument('--atk_factor',      default=None,               type=float,     help='Attack constant (sigma/p/scale)', nargs='+')

# Learning
p.add_argument('--epochs',          default=60,                 type=int,       help='Number of epochs')
p.add_argument('--lr',              default=0.003,              type=float,     help='Learning rate')
p.add_argument('--optimizer',       default='adam',             type=str,       help='optimizer', choices=['adam','sgd'])

# LIF neuron
p.add_argument('--timesteps',       default=20,                 type=int,       help='Number of time-step')
p.add_argument('--leak_mem',        default=0.99,               type=float,     help='Leak_mem')
p.add_argument('--def_threshold',   default=1.0,                type=float,     help='Membrane threshold')

# Visualization
p.add_argument('--log',             action='store_true',                        help='to print the output on terminal or to log file')
p.add_argument('--plot',            action='store_true',                        help='plot images')
p.add_argument('--plot_batch',      default=1,                  type=int,       help='batch to plot')
p.add_argument('--train_display',   default=1,                  type=int,       help='freq (in epochs) for printing training progress')
p.add_argument('--test_display',    default=10,                 type=int,       help='freq (in epochs) for evaluating model during training')
p.add_argument('--see_model',       action='store_true',                        help='see model structure')
p.add_argument('--info',            action='store_true',                        help='see training info')

# Dev tools
p.add_argument('--debug',           action='store_true',                        help='enable debugging mode')
p.add_argument('--first',           action='store_true',                        help='only debug first epoch and first ten batches')

global args
args = p.parse_args()

if args.first and (not args.debug):
    raise RuntimeError('You must run the --first command with --debug')

#--------------------------------------------------
# Setup
#--------------------------------------------------
run, f, config, trainloader, testloader, model, criterion, optimizer, scheduler, now, state = setup('train', args)

with run:
    # tell wandb to watch what the model gets up to
    wandb.watch(model, log='all', log_freq=10)

    #--------------------------------------------------
    # Train the using surrogate gradients
    #--------------------------------------------------
    f.write('********** {} training and evaluation **********'.format(config.model_type))
    max_acc = 0
    min_loss = 10
    wandb.run.summary['best_acc'] = max_acc
    wandb.run.summary['best_loss'] = min_loss

    for epoch in range(1, config.epochs+1):
        train_loss = AverageMeter()
        model.train()
        start_time = datetime.datetime.now()

        # print('epoch {} batches completed: '.format(epoch), end='')

        for batch_idx, (data, labels) in enumerate(trainloader):
                
            data = data.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(data)

            loss   = criterion(outputs, labels)
            train_loss.update(loss.item(), labels.size(0))

            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))

            loss.backward()
            optimizer.step()

            if args.debug:
                print('Epoch {}/{} | Training progress: {}% [{}/{}] '.format(epoch, config.epochs, round((batch_idx+1)/len(trainloader)*100, 2), batch_idx+1, len(trainloader)), end='\r', flush=True)

            if (args.first) and batch_idx == 9:
                break
        
        if args.debug:
            print('')

        # adjust_learning_rate(optimizer, epoch, config.epochs) # ? use adjust_learning_rate?
        scheduler.step()

        if args.debug or ((epoch) % args.train_display_freq == 0):
            duration = datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            f.write('Epoch {}/{} | Loss: {:.6f} | LR: {:.6f} [{}]'.format(epoch, config.epochs, train_loss.avg, optimizer.param_groups[0]['lr'], duration), terminal=True)
            wandb.log({'epoch': epoch, 'loss': train_loss.avg, 'lr': optimizer.param_groups[0]['lr'], 'train_duration_mins': (duration.seconds / 60)}, step=epoch)

            if train_loss.avg < min_loss:
                min_loss = train_loss.avg
                wandb.run.summary['best_loss'] = min_loss

        if epoch == 1:
            wandb.log({'accuracy': 0, 'max_acc': max_acc}, step=epoch)

        if args.debug or ((epoch) % args.test_display_freq == 0):
            max_acc = test('train', f, config, args, testloader, model, state, epoch, max_acc, start_time=start_time)

        if args.first and epoch == 0:
            break

f.write('Highest accuracy: {:.6f}'.format(max_acc), terminal=True)
f.write('Total script time: {}'.format(datetime.timedelta(days=(datetime.datetime.now() - now).days, seconds=(datetime.datetime.now() - now).seconds)), terminal=True)

sys.exit(0)