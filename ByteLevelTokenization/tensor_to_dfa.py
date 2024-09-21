from ByteLevelTokenization.fsa_to_tensor import Automata, drawGraph
import numpy as np 
from typing import Set, Dict, Optional, List, Callable, Union
import os

def tensor_to_dfa(language_tensor, wildcard_mat, mat, idx2word: Dict[int, str]) -> Automata:
    dfa = Automata()
    V,S,S = language_tensor.shape
    # 添加状态机状态集
    for i in range(S):
        dfa.states.add(i)
    # 设置初态
    dfa.setstartstate(0)
    # 使用mat的第二列来设置终态
    for i in range(len(mat)):
        if mat[i][1] == 1:
            dfa.addfinalstates(i)
    dfa.addfinalstates_label(dfa.finalstates,1)
    # print('idx2word:')
    # print(idx2word)
    for i in range(V):
        inp=idx2word[i]
        for j in range(S):
            for k in range(S):
                if language_tensor[i][j][k] == 1:
                    dfa.addtransition(j,k,inp)
    inp = '$'
    for i in range(wildcard_mat.shape[0]):
        for j in range(wildcard_mat.shape[1]):
            if wildcard_mat[i][j] == 1:
                dfa.addtransition(i,j,inp)
    # 重新排序
    dfa.transitions = {k:dfa.transitions[k] for k in sorted(dfa.transitions)}
    for k,v in dfa.transitions.items():
        dfa.transitions[k]={v_k:v[v_k] for v_k in sorted(v)}
    drawGraph(dfa,os.path.join('./result/pic','new_dfa.png'))
    return dfa 