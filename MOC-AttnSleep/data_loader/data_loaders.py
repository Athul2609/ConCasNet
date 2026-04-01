import torch
from torch.utils.data import Dataset
import os
import numpy as np

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset,order,level,mode="train"):
        super(LoadDataset_from_numpy, self).__init__()
        
        # Load all files at once (vectorized)
        X_list = [np.load(f)["x"] for f in np_dataset]
        y_list = [np.load(f)["y"] for f in np_dataset]
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        
        if mode=="train":
            # Vectorized filtering using boolean masks
            # Create masks for each condition
            mask_keep_as_0 = np.isin(y_train, order[level+1:])  # Labels to convert to 0
            mask_keep_as_1 = (y_train == order[level])           # Label to convert to 1
            mask_keep = mask_keep_as_0 | mask_keep_as_1         # Combined: keep these samples
            
            # Update labels
            y_train[mask_keep_as_0] = 0
            y_train[mask_keep_as_1] = 1
            
            # Apply mask to both X and y
            X_train = X_train[mask_keep]
            y_train = y_train[mask_keep]

        else:
            # Test mode: no filtering, just load all data
            pass

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size,order,level):
    train_dataset = LoadDataset_from_numpy(training_files,order,level)
    test_dataset = LoadDataset_from_numpy(subject_files,order,level)
    full_test_dataset= LoadDataset_from_numpy(subject_files,order,level,mode="test")
    full_train_dataset = LoadDataset_from_numpy(training_files,order,level,mode="test")
    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    # if np.any(all_ys==3):
    #     print(np.where(all_ys==3))
    # print(len(all_ys))
    # print(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    # print(counts)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    full_test_loader =  torch.utils.data.DataLoader(dataset=full_test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    
    full_train_loader= torch.utils.data.DataLoader(dataset=full_train_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)
    
    return train_loader, test_loader,full_test_loader,full_train_loader, counts
