'''
helper functions for experiment formulation
author: Yiqi Xie, Rui Fang
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
    '''read in a model and several directions in the parameter space, 
        create the grid for rays from this model towards those directions,
        put the results in state dict format and save them in a structured directory.

        designed to RUN IN LOCAL (turn cuda off)
    '''
    
    # the naming format for outputs
    dname_fmt='dirscan_direction_{idir}'
    fname_fmt='dirscan_direction_{idir}_point_{ipt}'

    # check if the output directiory already exists
    # this prevents overwriting
    if os.path.exists(destdir_path):
        raise ValueError('output directory already exists')
    else:
        os.mkdir(destdir_path)

    # create some help files recording the information of this scan
    shutil.copy2(direction_path, os.path.join(destdir_path, 'directions.npy'))
    np.save(os.path.join(destdir_path, 'distgrid.npy'), distgrid)

    # load the source model and the directions
    center, template = param_load_from_checkpoint(src_path, swa, cuda, template)
    random_directions = RandomDirections.from_npy(direction_path)

    for i in range(random_directions.n):

        print('preparing for the {}/{}-th direction...'\
                .format(i+1, random_directions.n))

        # generate a grid for one direction
        new_params = random_directions.spread(center, i, distgrid)

        # prepare the directory and corresponding namings
        dname = dname_fmt.format(idir=i)
        dpath = os.path.join(destdir_path, dname)
        os.mkdir(dpath)
        fname_list = [fname_fmt.format(idir=i, ipt=j) for j in range(len(distgrid))]
        fpath_list = [os.path.join(dpath, fname) for fname in fname_list]

        # save the grid
        param_batchsave_as_statedicts(template, new_params, fpath_list)

        print()

    return destdir_path



def perform_direction_scan(srcdir_path, destdir_path, 
                           model, dataset, dataset_path, 
                           cuda=True, update_buffers=True):
    '''read in the prepared direction scan files and the dataset, 
        make evaluation in order, 
        save the results (train_acc, train_loss, test_acc, test_loss) in .npy files.

        designed to RUN ON HUB (turn cuda on)
    '''
    
    # check if the output directiory already exists
    # this prevents overwriting
    if os.path.exists(destdir_path):
        raise ValueError('output directory already exists')
    else:
        os.mkdir(destdir_path)

    # create a help file
    with open(os.path.join(destdir_path, 'info.txt'), 'w') as file:
        file.write('train_acc, train_loss, test_acc, test_loss\n')
    
    # initializations
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
        
        # check grids of one direction and prepare them in order
        f_list = os.listdir(subdir_path)
        f_list = [fname for fname in f_list if fname.startswith('dirscan')]
        igrid_list = [int(fname.rpartition('_')[2]) for fname in f_list]
        fpath_list = [os.path.join(subdir_path, fname) for fname in f_list]
        ordered = sorted(zip(igrid_list, fpath_list), key=lambda x:x[0])
        igrid_list = [x[0] for x in ordered]
        fpath_list = [x[1] for x in ordered]
        print('order check: {}'.format(igrid_list), end='\n\n')
        
        # batch evaluation
        res_list = statedict_batchevaluate(fpath_list, evaluator, cuda, update_buffers, dest_path=None) # printout process
        res_list = np.array(res_list) # shape (ngrid, 4)
        
        # save the results
        i_direction = subdir_path.rpartition('_')[2]
        outpath = os.path.join(destdir_path, 'dirscan_direction_{idir}.npy'.format(idir=i_direction))
        np.save(outpath, res_list)
        
        print(end='\n\n')
    
    return destdir_path




def prepare_line_scan(swa_src_path, sgd_src_path, 
                      destdir_path, distgrid, 
                      cuda=False, template=None):
    '''run this in local'''
    
    if os.path.exists(destdir_path):
        raise ValueError('output directory already exists')
    else:
        os.mkdir(destdir_path)
    
    np.save(os.path.join(destdir_path, 'distgrid.npy'), distgrid)
    
    param_swa, template = param_load_from_checkpoint(swa_src_path, True, cuda, template)
    param_sgd, template = param_load_from_checkpoint(sgd_src_path, False, cuda, template)  
    
    line = Line.from_AB(param_swa, param_sgd)
    new_params = line.spread(distgrid)
    
    fname_list = ['linescan_point_{ipt}'.format(ipt=i) for i in range(len(distgrid))]
    fpath_list = [os.path.join(destdir_path, fname) for fname in fname_list]

    param_batchsave_as_statedicts(template, new_params, fpath_list)

    return destdir_path



def perform_line_scan(srcdir_path, destdir_path, 
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
    
    f_list = os.listdir(srcdir_path)
    f_list = [fname for fname in f_list if fname.startswith('linescan')]
    igrid_list = [int(fname.rpartition('_')[2]) for fname in f_list]
    fpath_list = [os.path.join(srcdir_path, fname) for fname in f_list]
    ordered = sorted(zip(igrid_list, fpath_list), key=lambda x:x[0])
    igrid_list = [x[0] for x in ordered]
    fpath_list = [x[1] for x in ordered]
        
    print('order check: {}'.format(igrid_list), end='\n\n')
        
    res_list = statedict_batchevaluate(fpath_list, evaluator, cuda, update_buffers, dest_path=None) # printout process
    res_list = np.array(res_list) # shape (ngrid, 4)
        
    outpath = os.path.join(destdir_path, 'linescan.npy')
    np.save(outpath, res_list)
    
    return destdir_path




def statedict_load(src_path, cuda=False):
    '''load single state dict from file'''
    
    configs = {'map_location': lambda storage, loc: storage} if not cuda else {}
    state_dict = torch.load(src_path, **configs)
    
    return state_dict



def statedict_evaluate(src_path, evaluator, cuda=True, update_buffers=True):
    '''load state dict from file and evluate'''
    
    state_dict = statedict_load(src_path, cuda)
    train_res, test_res = evaluator.evaluate(state_dict, update_buffers)
    
    return train_res['accuracy'], train_res['loss'], test_res['accuracy'], test_res['loss']



def statedict_batchevaluate(src_path_list, evaluator, cuda=True, update_buffers=True, dest_path=None):
    '''load a batch of state dicts and evaluate,
        by default the results will also be printed to screen'''
        
    columns = ['progress', 'train_acc', 'train_loss', 'test_acc', 'test_loss']
    header = ['{:>12s}'.format(c) for c in columns]
    header = ''.join(header) + '\n'
    
    if dest_path is not None:
        with open(dest_path, 'w') as file:
            file.write(header)
    else:
        sys.stdout.write(header)
    
    res_list = []
    
    for i, path in enumerate(src_path_list):
        train_acc, train_loss, test_acc, test_loss \
            = statedict_evaluate(path, evaluator, cuda, update_buffers)
        res = [train_acc, train_loss, test_acc, test_loss]
        res_list.append(res)
        
        prog = '{}/{}'.format(i+1, len(src_path_list))
        record = ['{:>12.4f}'.format(r) for r in res]
        record = '{:^12s}'.format(prog) + ''.join(record) + '\n'
        if dest_path is not None:
            with open(dest_path, 'a') as file:
                file.write(record)
        else:
            sys.stdout.write(record)
    
    return res_list
                  


def param_load_from_checkpoint(src_path, swa=False, cuda=False, template=None):
    '''load checkpoint and translate into numpy array, 
        will generate a StateDictWrapper template if not given'''
    
    key = 'state_dict' if not swa else 'swa_state_dict'
    configs = {'map_location': lambda storage, loc: storage} if not cuda else {}
    
    checkpt = torch.load(src_path, **configs)
    if template is None:
        template = StateDictWrapper()
        template.fit(checkpt[key], clear_buffers=True)
    param = template.transform_to_array(checkpt[key])
    
    return param, template
    

    
def param_batchload_from_checkpoints(src_path_list, swa=False, cuda=False, template=None):
    '''load batch checkpoints and translate into numpy arrays, 
        will generate a StateDictWrapper template if not given'''
    
    param_list = []
    for i, path in enumerate(src_path_list):
        print('\rloading and translating {}/{}...'\
                  .format(i+1, len(src_path_list)), end='')
        checkpt = torch.load(path, **configs)
        param, template = param_load_from_checkpoint(path, swa, cuda, template)
        param_list.append(p)
    print('done')
        
    return param_list, template



def param_save_as_statedict(template, param, dest_path, cuda=False):
    '''translate numpy array into state dict and save'''
    
    state_dict = template.transform_to_state_dict(param, cuda)
    torch.save(state_dict, dest_path)
    
    return



def param_batchsave_as_statedicts(template, param_list, dest_path_list, cuda=False):
    '''translate batch of numpy arrays into state dicts and save'''
    
    for i, (param, path) in enumerate(zip(param_list, dest_path_list)):
        print('\rtranslating and saving {}/{}...'\
                  .format(i+1, len(param_list)), end='')
        param_save_as_statedict(template, param, path, cuda)
    print('done')
        
    return