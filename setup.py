#############################################
#   @author:                                #
#############################################

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

import wandb

import sys
import os
import datetime
import numpy as np

from utils import *
from models import *

def setup(phase, args):
    #--------------------------------------------------
    # Initialize seed
    #--------------------------------------------------
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #--------------------------------------------------
    # Configuration parameters
    #--------------------------------------------------
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime('%y%m%d-%H%M%S')
    date = now.strftime('%y%m%d')

    try:
        os.mkdir(model_dir)
    except OSError:
        pass

    if phase == 'train':
        config = dict(
            # Model
            model_path          = None,
            conversion          = None,
            model_type          = args.model_type,
            architecture        = args.arch,
            kernel_size         = args.kernel_size,
            pretrained          = args.pretrained,
            # Dataset
            dataset             = dataset_cfg[args.dataset],
            batch_size          = args.batch_size,
            batch_size_test     = args.batch_size*2,
            img_size            = (img_sizes[args.dataset] if args.img_size == -1 else (args.img_size, args.img_size)),
            augment             = args.augment,
            attack              = (args.attack if args.attack else False),
            atk_factor          = (args.atk_factor if (args.atk_factor or args.atk_factor == 0) else False),
            # Learning
            epochs              = args.epochs,
            lr                  = args.lr,
            optimizer           = args.optimizer,
            # LIF neuron
            timesteps           = (args.timesteps if args.model_type == 'snn' else None),
            leak_mem            = (args.leak_mem if args.model_type == 'snn' else None),
            def_threshold       = (args.def_threshold if args.model_type == 'snn' else None),
            scaling_factor      = None,
            # Visualization
            plot_batch          = args.plot_batch,
            count_spikes        = None,
        )
    elif phase == 'test':
        model_path = args.model_path
        conversion = True if 'conversion' in model_path else False

        state = torch.load(model_path, map_location='cpu')
        old_config = state['config']

        model_type = ('snn' if conversion else old_config['model_type'])

        config = dict(
            # Model
            model_path          = model_path,
            conversion          = conversion,
            model_type          = model_type,
            architecture        = old_config['architecture'],
            kernel_size         = old_config['kernel_size'],
            pretrained          = None,
            # Dataset
            dataset             = old_config['dataset'],
            batch_size          = (args.batch_size if conversion else None),
            batch_size_test     = args.batch_size,
            img_size            = old_config['img_size'],
            augment             = None,
            attack              = (args.attack if args.attack else False),
            atk_factor          = (args.atk_factor if (args.atk_factor or args.atk_factor == 0) else False),
            # Learning
            epochs              = None,
            lr                  = None,
            optimizer           = None,
            # LIF neuron
            timesteps           = old_config['timesteps'],
            leak_mem            = old_config['leak_mem'],
            def_threshold       = old_config['def_threshold'],
            scaling_factor      = (args.scaling_factor if conversion else None),
            # Visualization
            plot_batch          = args.plot_batch,
            count_spikes        = args.count_spikes,
        )

    #--------------------------------------------------
    # Initialize wandb settings
    #--------------------------------------------------

    # Generate tags
    tags = []
    if (args.debug):
        tags += ['development']
    else:
        tags += ['production']
    if phase == 'test':
        if args.max_act:
            tags += ['activations']
        elif config['conversion']:
            tags += ['conversion']
        if config['count_spikes']:
            tags += ['count spikes']
    if config['attack']:
        tags += ['attack']

    # Start a run, tracking hyperparameters
    run = wandb.init(
        project=args.project,
        group=date,
        job_type=phase,
        reinit=True,
        tags=tags,
        force=True,
        config=config,
        mode=args.wandb_mode
    )

    # Model identifier
    identifier = createIdentifier((date, run.name, wandb.config.model_type, wandb.config.architecture, wandb.config.dataset['name'], args.file_name))
    wandb.config.update({'identifier': identifier})

    config = wandb.config

    # Print wrapper
    f = File(False)

    if (args.debug):
        f.write('------------ D E V E L O P M E N T   M O D E -------------', start='\n', end='\n\n')
    f.write('Run on time: {}'.format(now))
    f.write('Identifier: {}'.format(config.identifier))
    if phase == 'test':
        f.write('Pretrained {}: {}'.format(config.model_type.upper(), args.model_path))
        if config.conversion:
            f.write('==== Converting ANN -> SNN [layer-wise thresholding] ====')
    
    if args.info:
        f.write('=== [{}] CONFIGURATION ==='.format(run.name), start='\n')
        for key in config.keys():
            if key == 'dataset':
                f.write('\t {:20} : {}'.format(key, getattr(config, key)['name']))
            else:
                value = getattr(config, key)
                if value != None:
                    f.write('\t {:20} : {}'.format(key, value))

    #--------------------------------------------------
    # Load dataset
    #--------------------------------------------------
    if config.dataset['name'] == 'cifar100':
        normalize       = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

        if phase == 'train' or config.conversion:
            transform_train = transforms.Compose([transforms.RandomCrop(config.img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            train_dataset   = datasets.CIFAR100(root=config.dataset['path'], train=True, download=True, transform=transform_train)
            trainloader    = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])
        test_dataset    = datasets.CIFAR100(root=config.dataset['path'], train=False, download=True, transform=transform_test)
        testloader     = DataLoader(dataset=test_dataset, batch_size=config.batch_size_test, shuffle=False)
    
    elif config.dataset['name'] == 'cifar10':
        normalize       = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        if phase == 'train' or config.conversion:
            transform_train = transforms.Compose([transforms.RandomCrop(config.img_size, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            train_dataset   = datasets.CIFAR10(root=config.dataset['path'], train=True, download=True, transform=transform_train)
            trainloader    = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)

        transform_test  = transforms.Compose([transforms.ToTensor(), normalize])
        test_dataset    = datasets.CIFAR10(root=config.dataset['path'], train=False, download=True, transform=transform_test)
        testloader     = DataLoader(dataset=test_dataset, batch_size=config.batch_size_test, shuffle=False)
    else:
        raise RuntimeError("dataset not valid..")

    if phase == 'train' or config.conversion:
        f.write('loaded {} train split [{} samples]'.format(config.dataset['name'], (len(trainloader)*config.batch_size)))
    f.write('loaded {} test split [{} samples]'.format(config.dataset['name'], (len(testloader)*config.batch_size_test)))

    #--------------------------------------------------
    # Instantiate the model and optimizer
    #--------------------------------------------------
    if config.model_type == 'snn':
        model = SNN_VGG(config=config)
    elif config.model_type == 'ann':
        model = ANN_VGG(config=config)
    else:
        raise RuntimeError("architecture not valid..")
    
    # print(model)

    if torch.cuda.is_available():
        model = model.cuda()

    if args.see_model:
        f.write(model)

    if phase == 'test':
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)

        if config.conversion:
            # If thresholds present in loaded ANN file
            if (not args.reset_thresholds) and ('thresholds' in state.keys()) and (str(config.timesteps) in state['thresholds'].keys()):
                thresholds = state['thresholds'][str(config.timesteps)]
                f.write('Loaded layer thresholds ({}) from {}'.format(config.timesteps, args.model_path))
                model.threshold_update(scaling_factor=config.scaling_factor, thresholds=thresholds[:])
            else:
                thresholds = find_thresholds(f, trainloader, model, config.batch_size_test, config.timesteps)
                model.threshold_update(scaling_factor=config.scaling_factor, thresholds=thresholds[:])
                
                # Save the threhsolds in the ANN file
                if ('thresholds' not in state.keys()) or (not isinstance(state['thresholds'], dict)):
                    state['thresholds'] = {}
                state['thresholds'][str(config.timesteps)] = thresholds
                torch.save(state, args.model_path)
                f.write('Saved layer thresholds ({}) in {}'.format(config.timesteps, args.model_path))

    if phase == 'train':
        # Configure the loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=5e-4) # ? Should we use amsgrad and weight_decay for adam?
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4) # ? weight_decay 1e-4 or 5e-4?
        else:
            raise RuntimeError("optimizer not valid..")

        milestones = [int(milestone*config.epochs) for milestone in [0.5, 0.8]]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    #--------------------------------------------------
    # Prepare state objects
    #--------------------------------------------------

    # Prepare state to be saved with trained model
    if phase == 'train':
        state = {
            'config': config.as_dict()
        }

    if phase == 'train':
        return run, f, config, trainloader, testloader, model, criterion, optimizer, scheduler, now, state
    elif phase == 'test':
        return run, f, config, testloader, model, now