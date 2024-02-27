import numpy as np
import random
from itertools import compress
from torch.utils.data import Dataset
import torch
from scipy import signal


class EEGDatasetTrain(Dataset):
    def __init__(self, classIndices, ids, root, labels, len_sec, fc, balanceData=True, normalize=True, train=True,
                 classes=3):
        self.root = root
        self.labels = labels
        self.ids = ids
        self.datapoints = fc * len_sec
        self.classIdx = classIndices
        self.proc_samples = []
        self.newlabels = []
        self.pat_ids = []
        self.classes = classes

        self.process(normalize)

    def process(self, normalize):
        for index in range(0, len(self.ids)):
            path = self.root + "EEG_PREVIEW_" + str(self.ids[index]) + "_data.dat"
            f = open(path, 'r')  # opening a binary file
            data = np.fromfile(f, dtype=np.float32)
            data = np.reshape(data, (-1, 19))
            data = np.transpose(data)
            dps = int(self.datapoints)

            # normalization
            if normalize:
                m = np.mean(data, axis=1)
                dev_st = np.std(data, axis=1)
                data = (data - m.reshape(19, -1))/dev_st.reshape(19, -1)

            data = np.expand_dims(data, axis=0)


            i = 0
            while i < (data.shape[2] - dps):
                td = data[:, :, i: i + dps]
                self.newlabels.append(self.classIdx[self.labels[index]])
                self.proc_samples.append(td)
                self.pat_ids.append(self.ids[index])
                i = i + dps
            #print(i, str(self.ids[index]))

    def __len__(self):
        return len(self.proc_samples)

    def __getitem__(self, idx):
        label_i = self.newlabels[idx]
        to_return = torch.from_numpy(self.proc_samples[idx]).float()
        patient_i = self.pat_ids[idx]
        return to_return, torch.tensor(label_i), torch.tensor(patient_i)


class EEGDatasetTrain2(Dataset):
    def __init__(self, classIndices, ids, root, labels, len_sec, fc, balanceData=True, normalize=True, classes=3):
        self.root = root
        self.labels = labels
        self.ids = ids
        self.datapoints = fc * len_sec
        self.classIdx = classIndices
        self.proc_samples = []
        self.proc_samples2 = []
        self.newlabels = []
        self.pat_ids = []
        self.classes = classes

        self.process(normalize)
        if balanceData:
            self.balance()

    def process(self, normalize):
        for index in range(0, len(self.ids)):
            path = self.root + "EEG_PREVIEW_" + str(self.ids[index]) + "_data.dat"
            f = open(path, 'r')  # opening a binary file
            data = np.fromfile(f, dtype=np.float32)
            data = np.reshape(data, (-1, 19))
            data = np.transpose(data)
            data2 = data

            epochs = int(data.shape[1] / self.datapoints)

            # normalization
            if normalize:
                m = np.mean(data, axis=1)
                dev_st = np.std(data, axis=1)
                data = (data - m.reshape(19, -1))/dev_st.reshape(19, -1)

            data = np.expand_dims(data, axis=0)
            data2 = np.expand_dims(data2, axis=0)

            for i in range(epochs):
                td = data[:, :, i * self.datapoints:(i + 1) * self.datapoints]
                td2 = data2[:, :, i * self.datapoints:(i + 1) * self.datapoints]

                self.newlabels.append(self.classIdx[self.labels[index]])
                self.proc_samples.append(td)
                self.proc_samples2.append(td2)
                self.pat_ids.append(self.ids[index])

    def balance(self):
        a_num = np.sum(np.asarray(self.newlabels) == 0)
        b_num = np.sum(np.asarray(self.newlabels) == 1)

        a_indices = np.asarray(self.newlabels) == 0
        b_indices = np.asarray(self.newlabels) == 1

        a_list = list(compress(self.proc_samples, a_indices))
        a_labels = list(compress(self.newlabels, a_indices))
        a_pz = list(compress(self.pat_ids, a_indices))
        b_list = list(compress(self.proc_samples, b_indices))
        b_labels = list(compress(self.newlabels, b_indices))
        b_pz = list(compress(self.pat_ids, b_indices))

        min_val = np.min([a_num, b_num])

        random.seed(1)
        a_list = random.sample(a_list, min_val)
        random.seed(1)
        a_labels = random.sample(a_labels, min_val)
        random.seed(1)
        a_pz = random.sample(a_pz, min_val)
        random.seed(1)
        b_list = random.sample(b_list, min_val)
        random.seed(1)
        b_labels = random.sample(b_labels, min_val)
        random.seed(1)
        b_pz = random.sample(b_pz, min_val)

        self.newlabels = a_labels + b_labels
        self.proc_samples = a_list + b_list
        self.pat_ids = a_pz + b_pz

    def __len__(self):
        return len(self.proc_samples)

    def __getitem__(self, idx):
        label_i = self.newlabels[idx]
        to_return = torch.from_numpy(self.proc_samples[idx]).float()
        to_return2 = torch.from_numpy(self.proc_samples2[idx]).float()
        patient_i = self.pat_ids[idx]
        return to_return, to_return2, torch.tensor(label_i), torch.tensor(patient_i)


class MEPDataset(Dataset):
    def __init__(self, X, y, classIdx):
        self.timeseries = X
        self.labels = y
        self.classIdx = classIdx

    def __getitem__(self, index):
        features = self.timeseries[index]
        label = self.classIdx[self.labels[index]]
        return torch.tensor(features), torch.tensor(label)

    def __len__(self):
        return len(self.timeseries)
