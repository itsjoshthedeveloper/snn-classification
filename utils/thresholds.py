import torch
import torch.nn as nn

def find_thresholds(f, loader, model, batch_size, timesteps=20):
    thresholds = []
    
    def find(layer):
        
        max_act = 0

        for batch_idx, (data, target) in enumerate(loader):
            
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                model.eval()
                output = model(data, find_max_mem=True, max_mem_layer=layer)

                max_act = output.item()
                thresholds.append(max_act)
                model.threshold_update(scaling_factor=1.0, thresholds=thresholds[:])

                break

    for l in model.features.named_children():
        if isinstance(l[1], nn.Conv2d):
            find(int(l[0]))
            print('Finding features thresholds {}% [Layer {}/{}]'.format(round((int(l[0]) + 1) / len(model.features) * 100, 2), int(l[0]) + 1, len(model.features)), end='\r', flush=True)
    
    print()
    
    for c in model.classifier.named_children():
        if isinstance(c[1], nn.Conv2d) and int(c[0]) < (len(model.classifier)-1):
            find(int(l[0]) + int(c[0]) + 1)
            print('Finding classifier thresholds {}% [Layer {}/{}]'.format(round((int(c[0]) + 1) / len(model.classifier) * 100, 2), int(c[0]) + 1, len(model.classifier)), end='\r', flush=True)
            
    print()
    
    return thresholds