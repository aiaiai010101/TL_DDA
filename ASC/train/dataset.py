import torch
from torch.utils.data import Dataset
import numpy as np

class ASC_Dataset(Dataset):
    def __init__(self, path):
        super(Dataset, self).__init__()

        data = np.load(path)
        self.data = {}
        self.data_keys=[]  #为了固定数据顺序

        for key, value in data.items():
            self.data_keys.append(key)
            self.data[key] = torch.tensor(value)

        key_first=self.data_keys[0]
        self.len=self.data[key_first].shape[0]
        if self.len==642:
            self.len=638

    def __getitem__(self, index):
        #每一条数据由9部分组成：['sentence', 'label', 'sen_len', 'target_word', 'target_len',
        # 'position', 'attention_1', 'pos', 'syntax_position']
        # #index是个整数标量，return_value只包含一条数据
        # return_value = []
        return_value = []
        for key in self.data_keys:
            return_value.append(self.data[key][index])
        return return_value

    def __len__(self):
        return self.len