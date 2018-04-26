'''
helper functions for experiment formalization and analysis
author: Yiqi Xie
'''

import os
import sys
import shutil
import torch
import numpy as np

from torch_model import StateDictWrapper, ModelEvaluator
from parameter_space import Line, Plane, DimReducedMesh, RandomDirections



def prepare_direction_scan(src_path, destdir_path, 
                           direction_path, distgrid, 
                           dname_fmt='dirscan_dirction_{idir}',
                           fname_fmt='dirscan_direction_{idir}_point_{ipt}',
                           swa=False, cuda=False, template=None):

    if os.path.exists(destdir_path):
        raise ValueError('output directory already exists')
    else:
        os.mkdir(destdir_path)

    shutil.copy2(direction_path, os.path.join(destdir_path, 'directions.npy'))
    np.save(os.path.join(destdir_path, 'distgrid.npy'), distgrid)

    center, template = param_load_from_checkpoint(src_path, swa, cuda, template)
    random_directions = RandomDirections.from_npy(direction_path)

    for i in range(random_directions.n):

        print('preparing for the {}/{}-th direction...'\
                .format(i+1, random_directions.n))

        new_params = random_directions.spread(center, i, distgrid)

        dname = dname_fmt.format(idir=i)
        dpath = os.path.join(destdir_path, dname)
        os.mkdir(dpath)

        fname_list = [fname_fmt.format(idir=i, ipt=j) for j in range(len(distgrid))]
        fpath_list = [os.path.join(dpath, fname) for fname in fname_list]

        param_savebatch_as_statedicts(template, new_params, fpath_list)

        print()

    return destdir_path





def param_load_from_checkpoint(src_path, swa=False, cuda=False, template=None):
    
    key = 'state_dict' if not swa else 'swa_state_dict'
    configs = {'map_location': lambda storage, loc: storage} if not cuda else {}
    
    checkpt = torch.load(src_path, **configs)
    if template is None:
        template = StateDictWrapper()
        template.fit(checkpt[key], clear_buffers=True)
    param = template.transform_to_array(checkpt[key])
    
    return param, template
    

def param_batchload_from_checkpoints(src_path_list, swa=False, cuda=False, template=None):
    
    param_list = []
    for i, path in enumerate(src_path_list):
        print('\rloading and tranlating {}/{}...'.format(i+1, len(src_path_list)), end='')
        checkpt = torch.load(path, **configs)
        param, template = param_load_from_checkpoint(path, swa, cuda, template)
        param_list.append(p)
    print('done')
        
    return param_list, template


def param_save_as_statedict(template, param, dest_path):
    
    state_dict = template.transform_to_state_dict(param)
    torch.save(state_dict, dest_path)
    
    return


def param_savebatch_as_statedicts(template, param_list, dest_path_list):
    
    for i, (param, path) in enumerate(zip(param_list, dest_path_list)):
        print('\rtranslating and saving {}/{}...'.format(i+1, len(param_list)), end='')
        param_save_as_statedict(template, param, path)
    print('done')
        
    return


def statedict_load(src_path, cuda=False):
    
    configs = {'map_location': lambda storage, loc: storage} if not cuda else {}
    state_dict = torch.load(src_path, **configs)
    
    return state_dict