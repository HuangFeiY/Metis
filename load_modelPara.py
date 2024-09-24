import os
import torch
from models.BRNN_O import IntentIntegrateOnehot
import pickle
import numpy as np

def load_modelPara(model_path):
    with open(model_path,'rb') as f:
        origin_model = pickle.load(f)
    origin_tensor=origin_model.fsa_rnn.fsa_tensor
    origin_tensor=origin_tensor.detach().numpy()
    mat_tensor=origin_model.mat 
    mat_tensor=mat_tensor.detach().numpy()
    return origin_tensor, mat_tensor

def torchLoad_modelPara(model_path):
    model=torch.load(model_path)
    newTensor=model.fsa_rnn.fsa_tensor
    newTensor=newTensor.detach().numpy()
    new_mat_tensor=model.mat
    new_mat_tensor = new_mat_tensor.detach().numpy()
    return newTensor, new_mat_tensor
    



if __name__ == '__main__':
    model_path = './result/model/chat_origin_randInitial.pkl'
    fsa_tensor, mat_tensor = load_modelPara(model_path)
    with open('./result/parameter/randInitial_originModel_para_T.txt', 'w') as f:
        for i in range(fsa_tensor.shape[0]):
            np.savetxt(f, fsa_tensor[i], fmt='%.6f')
            f.write("\n")  # 在每个切片后添加换行






# print('new_mat_tensor:', new_mat_tensor)
# with open('./result/parameter/newModel_para_T.txt', 'w') as f:
#     for i in range(newTensor.shape[0]):
#         np.savetxt(f, newTensor[i], fmt='%.6f')
#         f.write("\n")  # 在每个切片后添加换行