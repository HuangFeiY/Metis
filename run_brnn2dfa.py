import argparse
import os 
import numpy as np
import pickle
from copy import deepcopy
from load_modelPara import load_modelPara
from ByteLevelTokenization.load_dataset import load_classification_dataset
from utils.utils import split_wildcard_mat
from ByteLevelTokenization.fsa_to_tensor import Automata
from ByteLevelTokenization.tensor_to_dfa import tensor_to_dfa

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='chat', help="dataset name")
    parser.add_argument('--dataset_spilt', type=float, default=1.0, help="rate of using labeled data, [0.01, 0.1, 1]")
    parser.add_argument('--model_dir', type=str, default='./result/model',help='model directory')
    parser.add_argument('--test_spilt', type=float, default=0.2, help="spilt rate of test set")
    parser.add_argument('--val_spilt', type=float, default=0.1, help="spilt rate of validation set")
    parser.add_argument('--save_path', type=str, default='./result/automata', help="save path of automata")
    parser.add_argument('--save_name', type=str, default='new_dfa.pkl', help="save file name of automata")
    
    args = parser.parse_args()
    args_bak = deepcopy(args)
    
    model_path=os.path.join(args.model_dir, '{}_origin.pkl'.format(args.dataset))
    complete_tensor_extend, mat = load_modelPara(model_path)
    print('complete_tensor_extend.shape:',complete_tensor_extend.shape)
    print('mat.shape:',mat.shape)
    
    complete_tensor = complete_tensor_extend[:-1, :, :]
    wildcard_mat=split_wildcard_mat(complete_tensor)
    
    # with open('./result/parameter/wildcard_mat.txt', 'w') as f:
    #     np.savetxt(f, wildcard_mat, fmt='%.6f')
    
    language_tensor = complete_tensor - wildcard_mat
    
    dset = load_classification_dataset(args)
    # t2i即wordToIndex，i2t即Index2Word
    t2i, i2t, in2i, i2in = dset['t2i'], dset['i2t'], dset['in2i'], dset['i2in']
    i2t[len(i2t)] = '<pad>'
    t2i['<pad>'] = len(i2t) - 1
    # print('word2idx:')
    # print(t2i)
    dfa = Automata()
    dfa = tensor_to_dfa(language_tensor, wildcard_mat, mat, i2t)
    dfa_dict = dfa.to_dict()
    
    print('New automata saved!')
    
    path = os.path.join(args.save_path, args.save_name)
    pickle.dump(dfa_dict, open(path, 'wb'))
    
    
    
    