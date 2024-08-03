import sys
sys.path.append('../')
from client.client_base import Client_base as Client
from cnn import FMNIST_CNN, CIFAR10_CNN
from torch.utils.data import Dataset
import torch
import copy
import torch.nn as nn
import torch.optim as opAccuracyim
from progress.bar import Bar
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
from torchvision.models import resnet18
from sampling import partition_data_various_alpha, partition_data_various_alpha_imagenet,LocalDataloaders
from utils import Accuracy,average_weights
import warnings
warnings.filterwarnings('ignore')
import json
from clustering import get_gradients
from sklearn.metrics import pairwise_distances
from itertools import product

class Server_DivFL(object):
    def __init__(self,args):
        self.args = args
        self.seed = args["seed"]
        self.device = args["device"][0]
        self.clients = []
        if args["dataset"] == "CIFAR10":
            self.global_model = CIFAR10_CNN().to(self.device)
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha(n_users = args["n_clients"], alphas=  args["alphas"], rand_seed =  args["seed"], dataset='CIFAR10')

            for m in self.global_model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
        elif args["dataset"] == "FMNIST":
            self.global_model = FMNIST_CNN().to(self.device)
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha(n_users = args["n_clients"], alphas=  args["alphas"], rand_seed =  args["seed"], dataset='FMNIST')
            for m in self.global_model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                    
        elif args["dataset"] == "Mini-ImageNet":
            self.global_model = resnet18(weights='IMAGENET1K_V1')
            for param in self.global_model.parameters():
                param.requires_grad = False
            # Randomly initialize and modify the model's last layer for mini-imagenet.
            self.global_model.fc = nn.Linear(self.global_model.fc.in_features, args["n_classes"])
            self.global_model = self.global_model.to(self.device)
            
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha_imagenet(n_users = args["n_clients"], alphas= args["alphas"], rand_seed = args["seed"])
            self.global_model.fc.weight = nn.init.normal_(self.global_model.fc.weight, mean=0.0, std=0.01)    
            self.global_model.fc.bias = nn.init.zeros_(self.global_model.fc.bias)
                
        Loaders_train = LocalDataloaders(train_dataset,dict_users,args["batch_size"],ShuffleorNot = True)
        Loaders_test = LocalDataloaders(testset,dict_users_test,args["batch_size"],ShuffleorNot = True)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"],shuffle=False, num_workers=2)
        self.ckpt_path = "./weights/"
        # create clients
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx,copy.deepcopy(self.global_model), Loaders_train[idx], Loaders_test[idx]))

        n_samples = np.array([len(client.trainloader.dataset) for client in  self.clients])
        self.weights = n_samples / np.sum(n_samples)
        self.Global_acc = []
        self.Local_acc = []
        self.Mean_loss = []

        self.gradients = get_gradients("clustered_2", self.global_model, [self.global_model] * args["n_clients"])
        
    def train(self):
        global_weights = self.global_model.state_dict()
        n_sampled = max(int(self.args["frac"] * self.args["n_clients"]), 1)
        
        for epoch in tqdm(range(self.args["epochs"])):
            previous_global_model = copy.deepcopy(self.global_model)
            print(f'\n | Global Training Round : {epoch+1} |\n')
            local_weights = []
            local_loss = []
            local_acc = []
            clients_models = []
            sampled_clients_for_grad = []
            print("divfl sampling")
            np.random.seed(epoch)
            random_pool = list(range(self.args["n_clients"]))
            sampled_clients = self.submod_sampling(epoch, self.gradients, n_sampled, stochastic = True)
            
            print("selection in epoch: ", epoch)
            print(sampled_clients) 

            for idx in range(self.args["n_clients"]):
                self.clients[idx].load_model(global_weights)
                w,loss = self.clients[idx].local_training()
                if idx in sampled_clients:
                    local_weights.append(copy.deepcopy(w))
                clients_models.append(copy.deepcopy(self.clients[idx].model))
                sampled_clients_for_grad.append(idx)

            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            self.global_acc()
            self.Local_acc.append(np.mean(local_acc))
            self.Mean_loss.append(np.mean(local_loss))

            gradients_i = get_gradients("clustered_2", previous_global_model, clients_models)
            for idx, gradient in zip(sampled_clients_for_grad, gradients_i):
                self.gradients[idx] = gradient
        
        stat = {}    
        stat["Global_acc"] = self.Global_acc
        stat["Local_acc"] = self.Local_acc
        stat["Mean_loss"] = self.Mean_loss
        torch.save(global_weights, self.ckpt_path + self.args["exp"] + ".pt")
        json.dump(stat, open(self.ckpt_path + self.args["exp"] + ".json", "w")) 

    def global_acc(self):
        accuracy = 0
        cnt = 0
        self.global_model.eval()
        for cnt, (X,y) in enumerate(self.test_loader):
            X = X.to(self.device)
            y = y.double().to(self.device)
            p = self.global_model(X)
            y_pred = p.argmax(1).double()
            accuracy += Accuracy(y,y_pred)
            cnt += 1
        print("accuracy of global test:",accuracy/cnt)
        self.Global_acc.append(accuracy/cnt)
        
    def submod_sampling(self, i, gradients,n_sampled,stochastic = False):
        norm_diff = self.compute_diff(gradients,"euclidean")
        np.fill_diagonal(norm_diff, 0)
        indices = self.select_cl_submod(i, num_clients=n_sampled, norm_diff = norm_diff, stochastic = stochastic)
        return indices

    def select_cl_submod(self,round, num_clients, norm_diff, stochastic = False):
        if stochastic:
            SUi = self.stochastic_greedy(norm_diff,num_clients)
        else:
            SUi = self.lazy_greedy(norm_diff,num_clients)
        indices = np.array(list(SUi))
        return indices

    def compute_diff(self,gradients, metric):  
        n_clients = len(gradients)
    
        metric_matrix = np.zeros((n_clients, n_clients))
        for i, j in product(range(n_clients), range(n_clients)):
            metric_matrix[i, j] = self.get_similarity(
                gradients[i], gradients[j], metric
            )
    
        return metric_matrix

    def get_similarity(self, grad_1, grad_2, distance_type="L1"):
        if distance_type == "L1":
            norm = 0
            for g_1, g_2 in zip(grad_1, grad_2):
                norm += np.sum(np.abs(g_1 - g_2))
            return norm
    
        elif distance_type == "euclidean":
            norm = 0
            for g_1, g_2 in zip(grad_1, grad_2):
                norm += np.sum((g_1 - g_2) ** 2)
            return np.sqrt(norm)
    
        elif distance_type == "cosine":
            norm, norm_1, norm_2 = 0, 0, 0
            for i in range(len(grad_1)):
                norm += np.sum(grad_1[i] * grad_2[i])
                norm_1 += np.sum(grad_1[i] ** 2)
                norm_2 += np.sum(grad_2[i] ** 2)
    
            if norm_1 == 0.0 or norm_2 == 0.0:
                return 0.0
            else:
                norm /= np.sqrt(norm_1 * norm_2)
    
                return np.arccos(norm)

    def lazy_greedy(self,norm_diff, num_clients):
            # initialize the ground set and the selected set
            V_set = set(range(self.args["n_clients"]))
            SUi = set()
    
            S_util = 0
            marg_util = norm_diff.sum(0)
            i = marg_util.argmin()
            L_s0 = 2. * marg_util.max()
            marg_util = L_s0 - marg_util
            client_min = norm_diff[:,i]
            # print(i)
            SUi.add(i)
            V_set.remove(i)
            S_util = marg_util[i]
            marg_util[i] = -1.
            
            while len(SUi) < num_clients:
                argsort_V = np.argsort(marg_util)[len(SUi):]
                for ni in range(len(argsort_V)):
                    i = argsort_V[-ni-1]
                    SUi.add(i)
                    client_min_i = np.minimum(client_min, norm_diff[:,i])
                    SUi_util = L_s0 - client_min_i.sum()
    
                    marg_util[i] = SUi_util - S_util
                    if ni > 0:
                        if marg_util[i] < marg_util[pre_i]:
                            if ni == len(argsort_V) - 1 or marg_util[pre_i] >= marg_util[argsort_V[-ni-2]]:
                                S_util += marg_util[pre_i]
                                # print(pre_i, L_s0 - S_util)
                                SUi.remove(i)
                                SUi.add(pre_i)
                                V_set.remove(pre_i)
                                marg_util[pre_i] = -1.
                                client_min = client_min_pre_i.copy()
                                break
                            else:
                                SUi.remove(i)
                        else:
                            if ni == len(argsort_V) - 1 or marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                                S_util = SUi_util
                                # print(i, L_s0 - S_util)
                                V_set.remove(i)
                                marg_util[i] = -1.
                                client_min = client_min_i.copy()
                                break
                            else:
                                pre_i = i
                                SUi.remove(i)
                                client_min_pre_i = client_min_i.copy()
                    else:
                        if marg_util[i] >= marg_util[argsort_V[-ni-2]]:
                            S_util = SUi_util
                            # print(i, L_s0 - S_util)
                            V_set.remove(i)
                            marg_util[i] = -1.
                            client_min = client_min_i.copy()
                            break
                        else:
                            pre_i = i
                            SUi.remove(i)
                            client_min_pre_i = client_min_i.copy()
            return SUi

    def stochastic_greedy(self,norm_diff, num_clients, subsample=0.1):
        # initialize the ground set and the selected set
        V_set = set(range(self.args["n_clients"]))
        SUi = set()

        m = max(num_clients, int(subsample * self.args["n_clients"]))
        for ni in range(num_clients):
            if m < len(V_set):
                R_set = np.random.choice(list(V_set), m, replace=False)
            else:
                R_set = list(V_set)
            if ni == 0:
                marg_util = norm_diff[:, R_set].sum(0)
                i = marg_util.argmin()
                client_min = norm_diff[:, R_set[i]]
            else:
                client_min_R = np.minimum(client_min[:,None], norm_diff[:,R_set])
                marg_util = client_min_R.sum(0)
                i = marg_util.argmin()
                client_min = client_min_R[:, i]
            SUi.add(R_set[i])
            V_set.remove(R_set[i])
        return SUi