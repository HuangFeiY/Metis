import os
import torch
from models.BRNN_O import IntentIntegrateOnehot
import pickle
import numpy as np

with open('./result/model/chat_origin.pkl','rb') as f:
    origin_model = pickle.load(f)
    
# print(origin_model.fsa_rnn.fsa_tensor.shape)
# print(origin_model.fsa_rnn.fsa_tensor)
origin_tensor=origin_model.fsa_rnn.fsa_tensor
origin_tensor=origin_tensor.detach().numpy()
mat_tensor=origin_model.mat 
print('mat_tensor:' , mat_tensor)
with open('./result/parameter/originModel_para_T.txt', 'w') as f:
    for i in range(origin_tensor.shape[0]):
        np.savetxt(f, origin_tensor[i], fmt='%.6f')
        f.write("\n")  # 在每个切片后添加换行


model=torch.load('./result/model/chat-1.0/chat-1.0.m')
newTensor=model.fsa_rnn.fsa_tensor
newTensor=newTensor.detach().numpy()
new_mat_tensor=model.mat
print('new_mat_tensor:', new_mat_tensor)
with open('./result/parameter/newModel_para_T.txt', 'w') as f:
    for i in range(newTensor.shape[0]):
        np.savetxt(f, newTensor[i], fmt='%.6f')
        f.write("\n")  # 在每个切片后添加换行