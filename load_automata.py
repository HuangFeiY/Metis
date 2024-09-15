import pickle
import os

dict_dir = "./mydata/snort/scan/automata"

# 从 pkl 文件加载字典
with open(os.path.join(dict_dir,'all.pkl'), 'rb') as f:
    loaded_dict = pickle.load(f)

print(loaded_dict)