#############################################
#   @author:                                #
#############################################

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.backends.cudnn as cudnn
import torchfunc

cudnn.enabled = False
cudnn.benchmark = True
cudnn.deterministic = True

import wandb

import argparse
import sys
import os
import datetime
import numpy as np
from matplotlib import pyplot as plt
from glob import glob

from utils import *
from setup import setup

def test(phase, f, config, args, testloader, model, state=None, epoch=0, max_acc=0, start_time=None, num_plot=16):

    if not start_time:
        start_time = datetime.datetime.now()

    if phase == 'test' and args.max_act:
        recorder = torchfunc.hooks.recorders.ForwardOutput()
        recorder.modules(model, types=(nn.Conv2d))
        print('created recorder')

    with torch.no_grad():
        model.eval()

        acc_top1, acc_top5 = [], []
        examples = None

        mem = 0
        num_scenarios = 0

        for batch_idx, (data, labels) in enumerate(testloader):

            if (args.debug or (phase == 'test' and args.max_act)) and (batch_idx + 1) != config.plot_batch:
                if phase == 'test':
                    f.write('Batch {} .................... skipped'.format(batch_idx + 1), end=('\r' if (batch_idx % 10) < 9 else '\n'), r_white=True, terminal=True)
            else:
                # batch_start = datetime.datetime.now()

                if torch.cuda.is_available():
                    data = data.cuda()
                    labels = labels.cuda()
                
                outputs  = model(data)

                prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
                acc_top1.append(float(prec1))
                acc_top5.append(float(prec5))

                if (batch_idx + 1) == config.plot_batch:
                    temp2 = {}
                    temp2['data'] = data.squeeze().cpu().numpy()
                    temp2['preds'] = outputs.max(1,keepdim=True)[1].squeeze().cpu().numpy()
                    temp2['labels']   = labels.squeeze().cpu().numpy()
                    examples = zip(temp2['data'][:num_plot], temp2['preds'][:num_plot], temp2['labels'][:num_plot])

                if phase == 'test':
                    f.write('Batch {} .................... completed'.format(batch_idx + 1), end=('\r' if (batch_idx % 10) < 9 else '\n'), r_white=True, terminal=True)
            
            if (args.debug or (phase == 'test' and args.max_act)) and (batch_idx + 1) == config.plot_batch:
                break

            if phase == 'test':
                print('Evaluating progress: {:05.2f}% [Batch {:04d}/{:04d}]'.format(round((batch_idx + 1) / len(testloader) * 100, 2), batch_idx + 1, len(testloader)), end='\r', flush=True)

    if phase == 'test' and args.max_act:
        plot_max_activations(f, args, recorder)

    if args.count_spikes:
        f.write('Average total spikes per example per layer: {}'.format(model.spikes.average()))
        f.write('Average neuronal spike rate per example per layer: {}'.format(model.spikes.rate()))
        f.write('Neurons per layer: {}'.format(model.spikes.units))

        f.write('Average total spikes per example: {}'.format(model.spikes.totalAverage()))
        f.write('Average neuronal spike rate per example: {}'.format(model.spikes.totalRate()))

        label, value, title = "layer", "total spikes per example", "Total Spikes Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.average())]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"total_spikes_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        value, title = "neuronal spike rate per example", "Neuronal Spike Rate Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.rate())]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"spike_rate_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        value, title = "neurons", "Neurons Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.units)]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"neurons_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        wandb.log({"total_spikes": model.spikes.totalAverage(), "spike_rate": model.spikes.totalRate(), "total_neurons": model.spikes.totalUnits()}, step=epoch)

    test_acc = np.mean(acc_top1)

    if test_acc > max_acc:
        max_acc = test_acc
        wandb.run.summary["best_acc"] = max_acc

        if (not args.debug) and phase == 'train':
            state = {
                **state,
                'max_acc'               : max_acc,
                'epoch'                 : epoch,
                'state_dict'            : model.state_dict(),
            }
            filename = model_dir+config.identifier+'.pth'
            torch.save(state, filename)
            
            filename = os.path.join(wandb.run.dir, config.identifier+'.pth')
            torch.save(state, filename)

            # filename = os.path.join(wandb.run.dir, config.identifier+'.onnx')
            # torch.onnx.export(model, data, filename, export_params=True, opset_version=11)
        
        if phase == 'train':
            identifier = 'examples'.format('epoch' + str(epoch))
        elif phase == 'test':
            if config.attack:
                identifier = '{}_{}_examples'.format(config.attack, config.atk_factor)
            else:
                identifier = '{}_examples'.format('batch' + str(config.plot_batch))

        # Plot examples
        if args.plot:
            cnt = 0
            columns = 4
            plt.figure(figsize=(15,((30/32)*num_plot)))
            for i, (image, pred, label) in enumerate(examples):
                cnt += 1
                plt.subplot(num_plot//columns,columns,cnt)
                plt.xticks([], [])
                plt.yticks([], [])

                if phase == 'test' and args.plot_labels:
                    plt.title("label: {}".format(config.dataset['labels'][label]), fontsize=8)

                image = image.transpose(1,2,0)
                plt.imshow(np.clip(image, 0, 1))

                plt.xlabel("pred: {}".format(config.dataset['labels'][pred]), fontsize=8)

            plt.suptitle('{}_{}'.format(config.identifier, identifier), fontsize=16)

            wandb.log({identifier: plt}, step=epoch)
        else:
            img_list = []
            for i, (image, pred, label) in enumerate(examples):
                image = image.transpose(1,2,0)
                caption = '{} [{}]'.format(config.dataset['labels'][pred], config.dataset['labels'][label])

                img_list.append(wandb.Image(image, caption=caption))

            wandb.log({identifier: img_list}, step=epoch)

    if phase == 'train':
        duration = datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        f.write('--------------- Evaluation -> accuracy: {:.3f}, best: {:.3f}, time: {}'.format(test_acc, max_acc, duration), terminal=True)
        wandb.log({'accuracy': test_acc, 'max_acc': max_acc, 'test_duration_mins': (duration.seconds / 60)}, step=epoch)

    return max_acc

if __name__ == '__main__':
    #--------------------------------------------------
    # Parse input arguments
    #--------------------------------------------------
    p = argparse.ArgumentParser(description='Evaluating ANN/SNN for classification', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Processing
    p.add_argument('--seed',            default=0,                  type=int,       help='Random seed')
    p.add_argument('--num_workers',     default=4,                  type=int,       help='number of workers')

    # Wandb and file
    p.add_argument('--wandb_mode',      default='online',           type=str,       help='wandb mode', choices=['online','offline','disabled'])
    p.add_argument('--project',         default='snn-classif',      type=str,       help='project name')
    p.add_argument('--file_name',       default='',                 type=str,       help='Add-on for the file name')

    # Model
    p.add_argument('--model_path',      default='',                 type=str,       help='pretrained model path')

    # Dataset
    p.add_argument('--batch_size',      default=128,                type=int,       help='Batch size')
    p.add_argument('--augment',         action='store_true',                        help='turn on data augmentation')
    p.add_argument('--attack',          default='',                 type=str,       help='adversarial attack', choices=['saltpepper','gaussiannoise'])
    p.add_argument('--atk_factor',      default=None,               type=float,     help='Attack constant (sigma/p/scale)')

    # LIF neuron
    p.add_argument('--scaling_factor',  default=0.7,                type=float,     help='scaling factor for thresholds')
    p.add_argument('--reset_thresholds',action='store_true',                        help='find new thresholds for this number of timesteps')

    # Visualization
    p.add_argument('--plot',            action='store_true',                        help='plot images')
    p.add_argument('--plot_batch',      default=1,                  type=int,       help='batch to plot')
    p.add_argument('--plot_labels',     default=True, const=True,   type=str2bool,  help='plot images with labels', nargs='?')
    p.add_argument('--max_act',         default='',                 type=str,       help='only get max activations', choices=['pixel-img','channel-norm','pixel-norm'])
    p.add_argument('--see_model',       action='store_true',                        help='see model structure')
    p.add_argument('--info',            default=True, const=True,   type=str2bool,  help='see training info', nargs='?')
    p.add_argument('--count_spikes',    action='store_true',                        help='count spikes')

    # Dev tools
    p.add_argument('--debug',           action='store_true',                        help='enable debugging mode')
    p.add_argument('--first',           action='store_true',                        help='only debug first epoch and first ten batches')
    p.add_argument('--print_models',    action='store_true',                        help='only print available trained models')

    global args
    args = p.parse_args()

    #--------------------------------------------------
    # Initialize arguments
    #--------------------------------------------------

    if args.augment and args.attack:
        raise RuntimeError('You can\'t use the --augment command with the --attack command')

    if args.attack and (not args.atk_factor):
        raise RuntimeError('You must provide an attack (sigma/p/scale) constant with the --attack command')

    if args.print_models and args.model_path:
        raise RuntimeError('You can\'t use the --model_path command with the --print_models command')

    if args.model_path and args.model_path.isdigit():
        args.model_path = int(args.model_path)

    if isinstance(args.model_path, str) and args.model_path:
        args.model_path = (model_dir + args.model_path)
    else:
        pretrained_models = sorted(glob(model_dir + '*.pth'))
        val = args.model_path
        if not val and val != 0:
            print('---- Trained models ----')
            for i, model in enumerate(pretrained_models):
                print('{}: {}'.format(i, model[17:]))
            if args.print_models:
                exit()
            val = int(input('\n Which model do you want to use? '))
            while (val < 0) or (val >= len(pretrained_models)):
                print('That index number is not accepted. Please input one of the index numbers above.')
                val = int(input('\n Which model do you want to use? '))
        args.model_path = pretrained_models[val]
    print(args.model_path)


    #--------------------------------------------------
    # Setup
    #--------------------------------------------------
    factor = 'no atk' if args.atk_factor == None else args.atk_factor
    if args.attack:
        args.file_name = args.attack + '-' + str(factor)

    run, f, config, testloader, model, now = setup('test', args)

    with run:
        #--------------------------------------------------
        # Evaluate the model
        #--------------------------------------------------
        f.write('********** ({}) {} evaluation **********'.format(factor, config.model_type.upper()))
        max_acc = test('test', f, config, args, testloader, model)

    duration = datetime.timedelta(days=(datetime.datetime.now() - now).days, seconds=(datetime.datetime.now() - now).seconds)
    f.write('({}) Accuracy: {:.6f}'.format(factor, max_acc), r_white=True, terminal=True)
    f.write('({}) Run time: {}'.format(factor, duration), terminal=True)

    sys.exit(0)