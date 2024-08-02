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
import GPR
from GPR import Kernel_GPR

class Server_FedCor(object):
    def __init__(self,args):
        self.args = args
        self.seed = args["seed"]
        self.device = args["device"][0]
        self.clients = []
        if args["dataset"] == "CIFAR10":
            self.global_model = CIFAR10_CNN().to(self.device)
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha(n_users = args["n_clients"], alphas=  args["alphas"], rand_seed =  args["seed"], dataset='CIFAR10')

        elif args["dataset"] == "FMNIST":
            self.global_model = FMNIST_CNN().to(self.device)
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha(n_users = args["n_clients"], alphas=  args["alphas"], rand_seed =  args["seed"], dataset='FMNIST')
            
        elif args["dataset"] == "Mini-ImageNet":
            self.global_model = resnet18(weights='IMAGENET1K_V1')
            for param in self.global_model.parameters():
                param.requires_grad = False
            # Randomly initialize and modify the model's last layer for mini-imagenet.
            self.global_model.fc = nn.Linear(self.global_model.fc.in_features, args["n_classes"])
            self.global_model = self.global_model.to(self.device)
            
            train_dataset,testset, dict_users, dict_users_test =  partition_data_various_alpha_imagenet(n_users = args["n_clients"], alphas= args["alphas"], rand_seed = args["seed"])
        
        for m in self.global_model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                
        Loaders_train = LocalDataloaders(train_dataset,dict_users,args["batch_size"],ShuffleorNot = True)
        Loaders_test = LocalDataloaders(testset,dict_users_test,args["batch_size"],ShuffleorNot = True)
        self.test_loader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"],shuffle=False, num_workers=2)
        self.ckpt_path = "./weights/"
        # create clients
        for idx in range(args["n_clients"]):
            self.clients.append(Client(args, idx,copy.deepcopy(self.global_model), Loaders_train[idx], Loaders_test[idx]))
        
        if args["kernel"]=='Poly':
             self.gpr = Kernel_GPR(args["n_clients"],loss_type=args["train_method"],reusable_history_length=args["group_size"],\
                              gamma=args["GPR_gamma"],device=self.device, dimension = args["dimension"],kernel=GPR.Poly_Kernel,\
                              order = 1,Normalize = args["poly_norm"])
        
        self.gpr.to(self.device)
        
        n_samples = np.array([len(client.trainloader.dataset) for client in  self.clients])
        self.weights = n_samples / np.sum(n_samples)
        self.Global_acc = []
        self.Local_acc = []
        self.Mean_loss = []
        
        
    def train(self):
        global_weights = self.global_model.state_dict()
        n_sampled = max(int(self.args["frac"] * self.args["n_clients"]), 1)
        gt_global_losses = []
        list_loss = []
        for idx in range(self.args["n_clients"]):
            self.clients[idx].load_model(global_weights)
            loss = self.clients[idx].train_loss()
            list_loss.append(loss) 
        gt_global_losses.append(list_loss)   

        for epoch in tqdm(range(self.args["epochs"])):
            print(f'\n | Global Training Round : {epoch+1} |\n')
            local_weights = []
            local_loss = []
            local_acc = []
            np.random.seed(epoch)
            
            if epoch > self.args["warmup"] and  epoch%self.args["GPR_interval"] != 0:
                sampled_clients = self.gpr.Select_Clients(n_sampled,self.args["epsilon_greedy"],\
                                                          self.weights, False,self.args["dynamic_TH"])
                print("GPR Chosen Clients:",sampled_clients)
            else:                       
                sampled_clients = np.random.choice(self.args["n_clients"], size=n_sampled, replace=False, p=self.weights)
                print("selection in epoch: ", sampled_clients)
        
            print("selection in epoch: ", epoch)
            print(sampled_clients) 
            
            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
                w, loss =  self.clients[idx].local_training()
                acc = self.clients[idx].local_accuracy()
                local_acc.append(acc)
                local_loss.append(loss)
                local_weights.append(copy.deepcopy(w))

            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            #--------------------------------------------------------------------------------------------------------
            list_loss = []
            for idx in range(self.args["n_clients"]):
                self.clients[idx].load_model(global_weights)
                loss = self.clients[idx].train_loss()
                list_loss.append(loss) 
            gt_global_losses.append(list_loss)   
            
            if epoch<=self.args["warmup"]:# warm-up
                self.gpr.Update_Training_Data([np.arange(self.args["n_clients"]),],\
                                              [np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),],epoch=epoch)

                if epoch == self.args["warmup"]:
                    print("Training GPR")
                    self.gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=1000,schedule_lr=False,update_mean=True,verbose=1)

            elif epoch>self.args["warmup"] and epoch%self.args["GPR_interval"]==0:
                self.gpr.Reset_Discount()
                print("Training with Random Selection For GPR Training:")

                self.gpr.Update_Training_Data([np.arange(self.args["n_clients"]),],\
                                              [np.array(gt_global_losses[-1])-np.array(gt_global_losses[-2]),],epoch=epoch)
                print("Training GPR")
                self.gpr.Train(lr = 1e-2,llr = 0.01,max_epoches=self.args["GPR_Epoch"],schedule_lr=False,\
                               update_mean=True,verbose=1)

            else:# normal and not optimization round
                self.gpr.Update_Discount(sampled_clients,self.args["discount"])
        
            #--------------------------------------------------------------------------------------------------------
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
