import numpy as np


def create_mat_and_bias(automata, in2i, i2in, model_type='Onehot'):
    # create
    # mat的大小应该是K*2，疑似对应论文中的alpha_0和alpha_1矩阵
    mat = np.zeros((len(automata['states']), len(i2in)))
    bias = np.zeros((len(i2in),))
    if model_type == 'FSARNN':
        bias[0] += 0.001

    # extract final states, for multiple final states, use OR gate
    for lab, states in automata['finalstates_label'].items():
        print(lab, " ", states)
        lab_idx = in2i[lab]
        for state in states:
            mat[state, lab_idx] = 1

    # print(i2in)
    # print(bias)

    return mat, bias

