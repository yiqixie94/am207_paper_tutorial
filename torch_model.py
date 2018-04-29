'''
combined from the former: 
    model_evaluator.py
    state_dict_wrapper.py
author: Rui Fang
'''
import os
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

import copy
from collections import OrderedDict

import models
import utils



class ModelEvaluator:
    
    def __init__(self, model, dataset, data_path='./data', batch_size=128, num_workers=4):
        
        # Set model 
        model_cfg = getattr(models, model)
        
        # Load the data_set
        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(data_path, dataset.lower())
        train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
        test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
        self.loaders = {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
        }
        num_classes = max(train_set.train_labels) + 1
        
        # Prepare model 
        self.model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        self.model.cuda()
        
        
    def evaluate(self, torch_state_dict, update_buffers=False):
        criterion = F.cross_entropy
        
        # Recover model from state_dict
        self.model.load_state_dict(torch_state_dict)
        
        # Update BatchNorm buffers (if any)
        if update_buffers:
            utils.bn_update(self.loaders['train'], self.model)
        
        # Evalute on the training and test sets
        train_res = utils.eval(self.loaders['train'], self.model, criterion)
        test_res = utils.eval(self.loaders['test'], self.model, criterion)
        
        return train_res, test_res



class StateDictWrapper:

    def __init__(self):
        self.template_state_dict = None
        self.param_shape = None
    
    def fit(self, state_dict, clear_buffers=False):
        """
        Obtain shapes of parameters in a given state_dict. Save as template (including buffers) for future transformation 
        from param_array to state_dict. If clear_buffers is true, all buffer values are set to zero. 
        """
        self.template_state_dict = copy.deepcopy(state_dict)
        self.param_shape = OrderedDict()
        
        for key, value in state_dict.items():
            if 'weight' in key or 'bias' in key:
                self.param_shape[key] = value.cpu().numpy().shape
            else:
                if clear_buffers:
                    self.template_state_dict[key][:] = 0
                else:
                    pass
                
    def transform_to_array(self, state_dict):
        """
        Transform state_dict to 1D param_array.
        """
        list_params = []
        
        for key, value in state_dict.items():
            if 'weight' in key or 'bias' in key:
                list_params.append(value.cpu().numpy().flatten())

        return np.concatenate(list_params)
      
    def transform_to_state_dict(self, param_array, cuda=True):
        """
        Transform 1D param_array to state_dict. 
        """
        for key, shape in self.param_shape.items():
            # self.template_state_dict[key] = torch.from_numpy(param_array[:np.prod(shape)].reshape(shape)).cuda()
            value = torch.from_numpy(param_array[:np.prod(shape)].reshape(shape))
            value = value.cuda() if cuda else value
            self.template_state_dict[key] = value
            param_array = param_array[np.prod(shape):]

        return copy.deepcopy(self.template_state_dict)


# Usage example 
# checkpoint = torch.load("./checkpoints/swa-checkpoint-125.pt", map_location=lambda storage, loc: storage)    # cpu mode 
# checkpoint = torch.load("./checkpoints/swa-checkpoint-125.pt")  # cuda mode 
# state_dict = checkpoint['swa_state_dict']
# wrapper = StateDictWrapper()
# wrapper.fit(state_dict, clear_buffers=True)
# param_array = wrapper.transform_to_array(state_dict)
# state_dict_new = wrapper.transform_to_state_dict(param_array)
        
