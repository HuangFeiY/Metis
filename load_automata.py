import pickle
import os

# dict_dir = "./data/snort/chat/automata"
dict_dir = "./result/automata"

# 从 pkl 文件加载字典
with open(os.path.join(dict_dir,'new_dfa.pkl'), 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)