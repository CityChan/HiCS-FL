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
from UCBsampler import UCBsampler
import math

class Server_Oort(object):
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

        self.n_samples = np.array([len(client.trainloader.dataset) for client in  self.clients])
        self.weights = self.n_samples / np.sum(self.n_samples)
        self.Global_acc = []
        self.Local_acc = []
        self.Mean_loss = []
        self.ucbSampler = UCBsampler(args)

        for idx in range(args["n_clients"]):
            feedbacks = {'reward': self.n_samples[idx], 'duration':1}
            self.ucbSampler.register_client(idx, feedbacks=feedbacks)
            self.ucbSampler.update_duration(idx, 1)
        
    def train(self):
        global_weights = self.global_model.state_dict()
        n_sampled = max(int(self.args["frac"] * self.args["n_clients"]), 1)
        for epoch in tqdm(range(self.args["epochs"])):
            print(f'\n | Global Training Round : {epoch+1} |\n')
            local_weights = []
            local_loss = []
            local_acc = []
            np.random.seed(epoch)
            feasible_clients = range(self.args["n_clients"])
            if epoch < 25:
                print("random sampling")
                sampled_clients = np.random.choice(self.args["n_clients"], size=n_sampled, replace=False, p=self.weights)

            else:
                print("ucb sampling")
                sampled_clients = self.ucbSampler.select_participant(n_sampled,feasible_clients)
                
            print("selection in epoch: ", epoch)
            print(sampled_clients) 
            
            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
                w, loss =  self.clients[idx].local_training()
                reward =  math.sqrt(loss)*self.n_samples[idx]
                feedbacks = {
                    'reward': reward,
                    'duration': 1,
                    'status': True,
                    'time_stamp': epoch
                }
                self.ucbSampler.update_client_util(idx, feedbacks=feedbacks)
                acc = self.clients[idx].local_accuracy()
                local_acc.append(acc)
                local_loss.append(loss)
                local_weights.append(copy.deepcopy(w))

            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            self.global_acc()
            self.Local_acc.append(np.mean(local_acc))
            self.Mean_loss.append(np.mean(local_loss))
            
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

    def persoanlized_test(self):
        self.global_model.load_state_dict(torch.load(self.ckpt_path + self.args["exp"] + ".pt"))
        global_weights = self.global_model.state_dict()
        Acc = 0
        for idx in range(self.args["n_clients"]):
            self.clients[idx].load_model(global_weights)
            w, loss =  self.clients[idx].local_training()
            acc = self.clients[idx].local_accuracy()
            Acc += acc
        print("the average personalized acc:")
        print(Acc/self.args["n_clients"])