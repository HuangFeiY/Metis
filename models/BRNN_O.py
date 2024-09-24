import torch
import torch.nn as nn


class BRNNIntegrateOnehot(nn.Module):
    def __init__(self, fsa_tensor=None, is_cuda=True, args=None):
        """
        Parameters
        ----------
        pretrained_embed: pretrained glove embedding,  V x D, numpy array
        trans_r_1: Tensor decomposition components 1, S x R (state x rank) numpy array
        trans_r_2: Tensor decomposition components 2, S x R (state x rank) numpy array
        embed_r: Tensor decomposition components 0, V x R (vocab size x R) numpy array
        config: config
        """
        super(BRNNIntegrateOnehot, self).__init__()

        # self.is_cuda = torch.cuda.is_available()
        self.is_cuda = torch.cuda.is_available() if is_cuda else is_cuda

        V, S, S = fsa_tensor.shape
        self.S = S
        self.h0 = self.hidden_init()  # S hidden state dim should be equal to the state dim
        
        #nn.Parameter: 是 PyTorch 中的一个类，用于标记张量是模型的一个参数。
        # nn.Parameter 会自动将其注册到模型的参数列表中，这样在调用优化器进行训练时，这些参数会被优化器管理和更新。
        # 那么状态转移矩阵，将作为模型参数的一部分，会随着模型训练而发生变化！
        if args==None:
            self.fsa_tensor = nn.Parameter(torch.from_numpy(fsa_tensor).float(), requires_grad=True)  # V x S x S
        elif args.randInitial:
            # 生成均值为 0.5，标准差为 1 的正态分布，并将其限制在 (0, 1) 范围内
            fsa_tensor = torch.normal(mean=0.5, std=1.0, size=(V, S, S))
            # self.fsa_tensor = nn.Parameter(torch.from_numpy(fsa_tensor).float(), requires_grad=True)  # V x S x S
            # nn.init.xavier_normal_(self.fsa_tensor)
            self.fsa_tensor = nn.Parameter(torch.clamp(fsa_tensor, min=0.0, max=1.0), requires_grad=True)
            # 生成值为 0 或 1 的随机初始化张量
            # fsa_tensor = torch.randint(0, 2, (V, S, S))  # 生成 0 或 1 的整数张量
            # self.fsa_tensor = nn.Parameter(fsa_tensor.float(), requires_grad=True)  # 转为 float 并设置为模型参数
        else:
            self.fsa_tensor = nn.Parameter(torch.from_numpy(fsa_tensor).float(), requires_grad=True)

    def forward(self, input, lengths):
        """
        unbatched version of forward.
        input: Sequence of Vectors in one sentence, matrix in B x L x D
        lengths: lengths vector in B

        https://towardsdatascience.com/taming-lstms-variable-sized-mini-\
        batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        explains what is packed sequence
        need to deal with mask lengths
        :return all hidden state B x L x S:
        """

        # 想要探究input是什么，B和L这两个维度数分别代表什么意思，见此代码文件110行
        # L相当于Length，B相当于Batch size，一个样本对应的是一个正则表达式。注意：每个样本的L应该是一样的，因为经过pad操作
        B, L = input.size()  # B x L
        hidden = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        # 对于一个长度为L的样本，应该有L个隐藏层向量，每个隐藏层向量有S维，一个Batch有B个样本，所以是B x L x S
        all_hidden = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        for i in range(L):
            inp = input[:, i]  # B
            # fsa_tensor的大小是V x S x S， B < V
            Tr = self.fsa_tensor[inp]  # B x S x S
            # 爱因斯坦求和约定 ，这里相当于矩阵乘法
            # L个隐藏层向量的迭代
            hidden = torch.einsum('bs,bsj->bj', hidden, Tr)  # B x R, B x R -> B x R
            
            
            # 新加尝试
            # hidden = torch.nn.functional.softmax(hidden, -1)
            # 新加的尝试
            hidden = torch.clamp(hidden, min=-10.0, max=10.0)  # 限制隐藏状态的数值范围
            
            all_hidden[:, i, :] = hidden

        return all_hidden

    def maxmul(self, hidden, transition):
        # 对隐藏层向量进行广播乘法
        temp = torch.einsum('bs,bsj->bsj', hidden, transition)
        max_val, _ = torch.max(temp, dim=1)
        return max_val

    def viterbi(self, input, lengths):
        """
        unbatched version of forward.
        input: Sequence of Vectors in one sentence, matrix in B x L x D
        lengths: lengths vector in B

        https://towardsdatascience.com/taming-lstms-variable-sized-mini-\
        batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        explains what is packed sequence
        need to deal with mask lengths
        :return all hidden state B x L x S:
        """
        B, L = input.size()  # B x L
        hidden = self.h0.unsqueeze(0).repeat(B, 1)  # B x S
        all_hidden = torch.zeros((B, L, self.S)).cuda() if self.is_cuda else torch.zeros((B, L, self.S))

        for i in range(L):
            inp = input[:, i]  # B
            Tr = self.fsa_tensor[inp]  # B x S x S
            hidden = self.maxmul(hidden, Tr)  # B x S,  B x S x S  -> B x S
            all_hidden[:, i, :] = hidden

        return all_hidden

    def hidden_init(self):
        hidden = torch.zeros((self.S), dtype=torch.float)
        hidden[0] = 1.0
        hidden = hidden.cuda() if self.is_cuda else hidden
        return hidden


class IntentIntegrateOnehot(nn.Module):
    def __init__(self, fsa_tensor, config=None,
                 mat=None, bias=None, is_cuda=True, args=None):
        # 直接写super(nn.Module) 是不正确的，因为 super() 的第一个参数应该是当前类，而不是父类。这样，Python 会沿着从当前类开始的继承链向上寻找合适的父类。
        super(IntentIntegrateOnehot, self).__init__()

        self.fsa_rnn = BRNNIntegrateOnehot(fsa_tensor, is_cuda, args)
        self.mat = nn.Parameter(torch.from_numpy(mat).float(), requires_grad=bool(config.train_linear))
        self.bias = nn.Parameter(torch.from_numpy(bias).float(), requires_grad=bool(config.train_linear))
        # 默认为0
        self.clamp_score = bool(config.clamp_score)
        # 默认为0
        self.clamp_hidden = bool(config.clamp_hidden)
        self.wfa_type = config.wfa_type

    def forward(self, input, lengths):
        if self.wfa_type == 'viterbi':
            out = self.fsa_rnn.viterbi(input, lengths)
        else:
            # input是什么，参见train_brnn.py的第374行
            # out相当于是all_hidden，即每个时间步的隐藏层向量
            out = self.fsa_rnn.forward(input, lengths)

        B, L = input.size()
        last_hidden = out[torch.arange(B), lengths - 1, :]  # select all last hidden
        if self.clamp_hidden:
            last_hidden = torch.clamp(last_hidden, 0, 1)
        scores = torch.matmul(last_hidden, self.mat) + self.bias
        if self.clamp_score:
            scores = torch.clamp(scores, 0, 1)
        return scores
