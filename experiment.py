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
                           swa=False, cuda=False, template=None):
    '''run this in local'''
    
    dname_fmt='dirscan_direction_{idir}'
    fname_fmt='dirscan_direction_{idir}_point_{ipt}'

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

        param_batchsave_as_statedicts(template, new_params, fpath_list)

        print()

    return destdir_path




def perform_direction_scan(srcdir_path, destdir_path, 
                           model, dataset, dataset_path, 
                           cuda=True, update_buffers=True):
    '''run this on hub'''
    
    if os.path.exists(destdir_path):
        raise ValueError('output directory already exists')
    else:
        os.mkdir(destdir_path)
        
    with open(os.path.join(destdir_path, 'info.txt'), 'w') as file:
        file.write('train_acc, train_loss, test_acc, test_loss\n')
        
    evaluator = ModelEvaluator(model, dataset, dataset_path)
    
    subdir_list = os.listdir(srcdir_path)
    subdir_list = [dname for dname in subdir_list if dname.startswith('dirscan')]
    subdir_path_list = [os.path.join(srcdir_path, dname) for dname in subdir_list]
    subdir_path_list = [path for path in subdir_path_list if os.path.isdir(path)]
    
    for i, subdir_path in enumerate(subdir_path_list):
        # dname_fmt == "dirscan_direction_{idir}"
        # fname_fmt == "dirscan_direction_{idir}_point_{ipt}"
        
        print('processing the {}/{}-th direction...'\
                  .format(i+1, len(subdir_path_list)))
        
        f_list = os.listdir(subdir_path)
        f_list = [fname for fname in f_list if fname.startswith('dirscan')]
        igrid_list = [int(fname.rpartition('_')[2]) for fname in f_list]
        fpath_list = [os.path.join(subdir_path, fname) for fname in f_list]
        
        res_list = statedict_batchevaluate(fpath_list, evaluator, cuda, update_buffers)
        res_list = [res_list[i] for i in igrid_list]
        res_list = np.array(res_list) # shape (ngrid, 4)
        
        i_direction = subdir_path.rpartition('_')[2]
        outpath = os.path.join(destdir_path, 'dirscan_direction_{idir}.npy'.format(idir=i_direction))
        np.save(outpath, res_list)
        
        print()
    
    return destdir_path




def statedict_load(src_path, cuda=False):
    
    configs = {'map_location': lambda storage, loc: storage} if not cuda else {}
    state_dict = torch.load(src_path, **configs)
    
    return state_dict


def statedict_evaluate(src_path, evaluator, cuda=True, update_buffers=True):
    
    state_dict = statedict_load(src_path, cuda)
    train_res, test_res = evaluator.evaluate(state_dict, update_buffers)
    
    return train_res['accuracy'], train_res['loss'], test_res['accuracy'], test_res['loss']


def statedict_batchevaluate(src_path_list, evaluator, cuda=True, update_buffers=True):
    
    res_list = []
    for i, path in enumerate(src_path_list):
        print('\rloading and evaluating {}/{}...'\
                 .format(i+1, len(src_path_list)), end='')
        train_acc, train_loss, test_acc, test_loss \
            = statedict_evaluate(path, evaluator, cuda, update_buffers)
        res_list.append((train_acc, train_loss, test_acc, test_loss))
    print('done')
    
    return res_list
                  


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
        print('\rloading and tranlating {}/{}...'\
                  .format(i+1, len(src_path_list)), end='')
        checkpt = torch.load(path, **configs)
        param, template = param_load_from_checkpoint(path, swa, cuda, template)
        param_list.append(p)
    print('done')
        
    return param_list, template


def param_save_as_statedict(template, param, dest_path, cuda=False):
    
    state_dict = template.transform_to_state_dict(param, cuda)
    torch.save(state_dict, dest_path)
    
    return


def param_batchsave_as_statedicts(template, param_list, dest_path_list, cuda=False):
    
    for i, (param, path) in enumerate(zip(param_list, dest_path_list)):
        print('\rtranslating and saving {}/{}...'\
                  .format(i+1, len(param_list)), end='')
        param_save_as_statedict(template, param, path, cuda)
    print('done')
        
    return