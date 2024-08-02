import numpy as np
import torch
import scipy
from torch.utils.data import Dataset
import torch
import copy
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pytorch_lmdb_imagenet.folder2lmdb  import ImageFolderLMDB

class LocalDataset(Dataset):
    """
    because torch.dataloader need override __getitem__() to iterate by index
    this class is map the index to local dataloader into the whole dataloader
    """
    def __init__(self, dataset, Dict):
        self.dataset = dataset
        self.idxs = [int(i) for i in Dict]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        X, y = self.dataset[self.idxs[item]]
        return X, y
    
def LocalDataloaders(dataset, dict_users, batch_size, ShuffleorNot = True):
    """
    dataset: the same dataset object
    dict_users: dictionary of index of each local model
    batch_size: batch size for each dataloader
    ShuffleorNot: Shuffle or Not
    """
    num_users = len(dict_users)
    loaders = []
    for i in range(num_users):
        loader = torch.utils.data.DataLoader(
                        LocalDataset(dataset,dict_users[i]),
                        batch_size=batch_size,
                        shuffle = ShuffleorNot,
                        num_workers=0,
                        drop_last=False)
        loaders.append(loader)
    return loaders


def partition_data_various_alpha(n_users, alphas=None,rand_seed = 0, dataset = 'CIFAR10'):
    if dataset == 'CIFAR10':
        K = 10
        data_dir = '../data/cifar10/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                          transform=apply_transform)
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
        
   
    if dataset == 'FMNIST':
        K = 10
        data_dir = '../data/FMNIST/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    
    N_parts = len(alphas)
    parts_users = int(n_users/N_parts)
    N = int(len(train_dataset)/N_parts)
    
    N_test = int(len(test_dataset)/N_parts)
    
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)
    

    for i in range(N_parts):
        alpha = alphas[i]
        min_size = 0
        
        while min_size < int(0.2*N/parts_users) or min_size_test < int(0.2*N_test/parts_users):
            idx_batch = [[] for _ in range(parts_users)]
            idx_batch_test = [[] for _ in range(parts_users)]
            for k in range(K):
                
                idx_k = np.where(y_train == k)[0][i*int(N/K):(i+1)*int(N/K)]
                idx_k_test = np.where(y_test == k)[0][i*int(N_test/K):(i+1)*int(N_test/K)]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, parts_users))
                ## Balance
                proportions_train = np.array([p*(len(idx_j)<N/parts_users) for p,idx_j in zip(proportions,idx_batch)])

                proportions_test = np.array([p*(len(idx_j)<N_test/parts_users) for p,idx_j in zip(proportions,idx_batch_test)])

                proportions_train = proportions_train/proportions_train.sum()
                proportions_test = proportions_test/proportions_test.sum()
                proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])

        for j in range(parts_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[i*parts_users +j] = idx_batch[j]
            net_dataidx_map_test[i*parts_users +j] = idx_batch_test[j]

        
    return (train_dataset, test_dataset,net_dataidx_map, net_dataidx_map_test)




def partition_data_various_alpha_imagenet(n_users, alphas=None,rand_seed = 0):
   
    K = 100
    data_dir = './data/imagenet/'
    train_path = data_dir + 'train.lmdb'
    test_path = data_dir + 'test.lmdb'
    data_transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
    train_loader = ImageFolderLMDB(train_path, data_transform)
    test_loader= ImageFolderLMDB(test_path, data_transform)

        
    y_train = np.load(data_dir + 'train_labels.npy')
    y_test = np.load(data_dir + 'test_labels.npy')

    
    N_parts = len(alphas)
    parts_users = int(n_users/N_parts)
    N = int(len(train_loader)/N_parts)
    
    N_test = int(len(test_loader)/N_parts)
    
    net_dataidx_map = {}
    net_dataidx_map_test = {}
    np.random.seed(rand_seed)
    

    for i in range(N_parts):
        alpha = alphas[i]
        min_size = 0
        
        while min_size < int(0.2*N/parts_users) or min_size_test < int(0.2*N_test/parts_users):
            idx_batch = [[] for _ in range(parts_users)]
            idx_batch_test = [[] for _ in range(parts_users)]
            for k in range(K):
                
                idx_k = np.where(y_train == k)[0][i*int(N/K):(i+1)*int(N/K)]
                idx_k_test = np.where(y_test == k)[0][i*int(N_test/K):(i+1)*int(N_test/K)]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, parts_users))
                ## Balance
                proportions_train = np.array([p*(len(idx_j)<N/parts_users) for p,idx_j in zip(proportions,idx_batch)])

                proportions_test = np.array([p*(len(idx_j)<N_test/parts_users) for p,idx_j in zip(proportions,idx_batch_test)])

                proportions_train = proportions_train/proportions_train.sum()
                proportions_test = proportions_test/proportions_test.sum()
                proportions_train = (np.cumsum(proportions_train)*len(idx_k)).astype(int)[:-1]
                proportions_test = (np.cumsum(proportions_test)*len(idx_k_test)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions_train))]
                idx_batch_test = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch_test,np.split(idx_k_test,proportions_test))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                min_size_test = min([len(idx_j) for idx_j in idx_batch_test])

        for j in range(parts_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[i*parts_users +j] = idx_batch[j]
            net_dataidx_map_test[i*parts_users +j] = idx_batch_test[j]

        
    return (train_loader, test_loader,net_dataidx_map, net_dataidx_map_test)