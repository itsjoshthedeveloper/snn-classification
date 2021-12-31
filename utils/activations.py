from matplotlib import pyplot as plt
from math import ceil
import wandb
import numpy as np


def plot_max_activations(f, args, recorder):
    activations = recorder.data
    f.write('\nNumber of subrecorders: {}'.format(len(activations)))
    f.write('Num of subrecorder batches: {}'.format([len(subrecorder) for subrecorder in activations]))
    f.write('Subrecorder batch {}: {}'.format(0, [subrecorder[0].shape for subrecorder in activations]))

    avg_mean, avg_min = 0, 0
    layers = [int(l) for l in input('Layers (a,b,c,d): ').split(',')]

    cols = 6 if len(layers) > 6 else len(layers)
    rows = int(ceil(len(layers)/cols))
    new_ratio = [int(val) for val in input('Aspect ratio [{},{}] (cols,rows): '.format(cols, rows)).split(',')]
    if new_ratio:
        cols, rows = new_ratio

    fig = plt.figure(figsize=(3.5*cols, 2.5*rows))

    subplot = 1
    for i, layer in enumerate(activations):
        if (i+1) in layers:
            layer = layer[0].cpu().numpy()
            
            max_activations = []

            if 'channel' in args.max_act:
                layer = layer.transpose(1, 0, 2, 3)
                for channel in layer:
                    channel_max_activations = []
                    for sample in channel:
                        channel_max_activations.append(np.max(sample))
                    max_activations.append(np.mean(channel_max_activations))
            elif 'pixel' in args.max_act and 'img' not in args.max_act:
                layer = layer.transpose(2, 3, 0, 1)
                for row in layer:
                    for pixel in row:
                        pixel_max_activations = []
                        for sample in pixel:
                            pixel_max_activations.append(np.max(sample))
                        max_activations.append(np.mean(pixel_max_activations))
            elif 'pixel-img' in 'pixel-img':
                max_activations = np.empty_like(layer[0][0])
                layer = layer.transpose(2, 3, 0, 1)
                for k, row in enumerate(layer):
                    for l, pixel in enumerate(row):
                        pixel_max_activations = []
                        for sample in pixel:
                            pixel_max_activations.append(np.max(sample))
                        max_activations[k][l] = np.mean(pixel_max_activations)

            f.write('---------------------------------\n')
            f.write('Subrecorder {}: {}'.format(i + 1, layer.shape))
            f.write('\n---------------------------------')

            ax = plt.subplot(rows, cols, subplot)
            subplot += 1
            ax.set_title('Conv {}'.format(i + 1))

            if 'img' in args.max_act:
                ax.axis('off')
                plt.imshow(max_activations)
            else:
                if 'norm' in args.max_act:
                    max_activations = (max_activations - np.min(max_activations))/np.ptp(max_activations)

                length = len(max_activations)
                f.write('\tNumber of {}: {}'.format(('channels' if 'channel' in args.max_act else 'pixels'), length))

                x = range(1, length + 1)
                mean = np.mean(max_activations)
                minimum = np.min(max_activations)
                # std = np.std(max_activations)
                f.write('\tMean: {}'.format(mean))
                f.write('\tMinimum: {}'.format(minimum))
                # f.write('\tStandard Deviation: {}\n'.format(std))
                
                markerline, stemlines, baseline = plt.stem(x, max_activations, 'xkcd:grey', 'o', 'k', use_line_collection=True)
                # plt.setp(markerline, color=plt.getp(stemlines, 'color')[0])
                markerline.set_markerfacecolor('k')
                markerline.set_markeredgecolor(plt.getp(stemlines, 'color')[0])
                markerline.set_markersize(3)

                mean_line = [mean] * length
                ax.plot(x, mean_line, 'b-', label='Mean', linewidth=3.0)

                min_line = [minimum] * length
                ax.plot(x, min_line, 'r-', label='Minimum', linewidth=3.0)

                # top_std = [mean + std] * length
                # bot_std = [mean - std] * length
                # ax.plot(x, top_std, 'C9-', label='Std Dev = {}'.format(round(float(std), 3)))
                # ax.plot(x, bot_std, 'C9-')
                
                leg = ax.legend(loc='upper right')

                avg_mean += mean
                avg_min += minimum
                # avg_std += std

    file_name = 'deeplab_max-activations_{}'.format(args.max_act)
    new_name = input('Filename [{}]: '.format(file_name))
    file_name = new_name if new_name else file_name
    
    if args.plot:
        wandb.log({'{}_activations'.format(args.max_act): plt})
        f.write('plotted and saved max activations', terminal=True)
    else:
        fig.canvas.set_window_title(file_name)
        plt.tight_layout()
        plt.show()

    f.write('Average mean: {}'.format(round(float(avg_mean/len(activations)), 3)), terminal=True)
    f.write('Average minimum: {}'.format(round(float(avg_min/len(activations)), 3)), terminal=True)
    # f.write('Average standard dev: {}'.format(round(float(avg_std/len(activations)), 3)), terminal=True)

    exit()