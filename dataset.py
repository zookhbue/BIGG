import math
import os
import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(
        self, filename, args, sort=False, drop_last=False
    ):
        # filename has three columns:  empirical_time-series_path,  coarse_time-series_path, label_path
        self.model = args.model
        self.empirical_path, self.coarse_path, self.label_true = self.process_meta(filename)

    def __len__(self):
        return len(self.empirical_path)

    def __getitem__(self, idx):
        path1 = self.empirical_path[idx]
        path2 = self.coarse_path[idx]
        
        empirical = np.loadtxt(path1,dtype=np.float32) #[187,90]

        timeseries = empirical.T  # left shape = [90, 187]
        timeseries_m = np.mean(timeseries, 1)
        timeseriesT = timeseries.T - timeseries_m
        timeseries1 = timeseriesT.T
        maxvalue = np.max(abs(timeseries1))
        timeseries1 = timeseries1 / maxvalue
        empirical = timeseries1.T


        coarse = np.loadtxt(path2,dtype=np.float32) # [90,187]
        coarse = coarse.T

        label_true = self.label_true[idx]

        sample = {
            "mel": empirical,
            "coarse": coarse,
            "label_true": label_true,
        }
        return sample
        
    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            path1 = []
            path2 = []
            label = []
            for line in f.readlines():
                # print('111111111111', line)
                columns = line.strip().split()
                # print(columns)
                path1.append(columns[0])
                path2.append(columns[1])
                label.append(int(columns[2]))
            # print('path1:  ', path1)
            # print('path2 ', path2)
            return path1,path2,label
