from random import random
import torch
import numpy as np
from torch.utils.data import Dataset


class FeatureDataset1(Dataset):
    def __init__(self, t_file, i_file,l_file,flag):
        if flag == "train":
            test_text = np.load(t_file)
            self.test_data_text = torch.from_numpy(test_text).float()
            test_img = np.load(i_file)
            self.test_data_img = torch.from_numpy(test_img).squeeze().float()
            test_labels = np.load(l_file)
            self.test_labels = torch.from_numpy(test_labels).long()
        elif flag == "test":
            test_text = np.load(t_file)
            self.test_data_text = torch.from_numpy(test_text).float()
            test_img = np.load(i_file)
            self.test_data_img = torch.from_numpy(test_img).squeeze().float()
            test_labels = np.load(l_file)
            self.test_labels = torch.from_numpy(test_labels).long()

    def __len__(self):
        return self.test_data_text.shape[0]

    def __getitem__(self, item):
        return self.test_data_text[item], self.test_data_img[item], self.test_labels[item]

