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
from clustering import get_gradients_fc,get_matrix_similarity_from_grads,get_clusters_with_alg2,get_similarity
from scipy.cluster.hierarchy import linkage
import scipy.stats
from itertools import product
from sklearn.cluster import AgglomerativeClustering 
from numpy.random import choice

class Server_Oracle(object):
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

        self.Counts = []
        for idx in range(args["n_clients"]):
            counts = [0]*args["n_classes"]
            for batch_idx,(X,y) in enumerate(Loaders_train[idx]):
                batch = len(y)
                y = np.array(y)
                for i in range(batch):
                    counts[int(y[i])] += 1
            self.Counts.append(np.array(counts))
        self.Entropy = self.compute_entropy()
        self.gradients = get_gradients_fc("clustered_2", self.global_model, [self.global_model] * args["n_clients"])

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
            sampled_clients = self.cluster_sampling(self.gradients, "cosine",epoch)
                
            print("selection in epoch: ", epoch)
            print(sampled_clients) 
            
            for idx in sampled_clients:
                self.clients[idx].load_model(global_weights)
                w, loss =  self.clients[idx].local_training()
                clients_models.append(copy.deepcopy(self.clients[idx].model))
                sampled_clients_for_grad.append(idx)
                acc = self.clients[idx].local_accuracy()
                local_acc.append(acc)
                local_loss.append(loss)
                local_weights.append(copy.deepcopy(w))

            global_weights = average_weights(local_weights)
            self.global_model.load_state_dict(global_weights)
            self.global_acc()
            self.Local_acc.append(np.mean(local_acc))
            self.Mean_loss.append(np.mean(local_loss))

            gradients_i = get_gradients_fc("clustered_2", previous_global_model, clients_models)
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

    def cluster_sampling(self,gradients, sim_type,epoch):
        Clusters = []
        n_sampled = max(int(self.args["frac"] * self.args["n_clients"]), 1)
        sim_matrix = self.get_matrix_similarity_from_grads_entropy(gradients, self.Entropy, distance_type=sim_type)
        linkage_matrix = linkage(sim_matrix, "ward") 
        hc = AgglomerativeClustering(n_clusters = self.args["M"], metric = "euclidean", linkage = 'ward') 
     
        hc.fit_predict(sim_matrix)
        labels = hc.labels_
        for i in range(self.args["M"]):
            cluster_i = np.where(labels == i)[0]
            Clusters.append(cluster_i)    
        avg_entropy = self.estimated_entropy(Clusters)
        
        return self.sample_clients_entropy(avg_entropy, Clusters,self.n_samples,epoch)


    def compute_entropy(self):
        Entropy = []
        for idx in range(self.args["n_clients"]):
            count = self.Counts[idx]
            pk = count/np.sum(count)
            entropy = scipy.stats.entropy(pk)
            Entropy.append(entropy)
        return Entropy

    def estimated_entropy(self, Clusters):
        Entropys = []
        for k in range(len(Clusters)):
            print("cluster ", k)
            print(Clusters[k])
            group_entropy = 0
            cluster = Clusters[k]
            for idx in cluster:
                group_entropy += self.Entropy[idx]
            if len(cluster) > 0:
                group_entropy = group_entropy/len(cluster)
            Entropys.append(group_entropy)
        Entropys = np.array(Entropys)
        return Entropys
    
    def get_matrix_similarity_from_grads_entropy(self, local_model_grads, entropy,distance_type):
        
        n_clients = len(local_model_grads)
        metric_matrix = np.zeros((n_clients, n_clients))
        for i, j in product(range(n_clients), range(n_clients)):
            metric = get_similarity(local_model_grads[i], local_model_grads[j], distance_type) 
            metric_matrix[i, j] =  metric + self.args["lambda"]*abs(entropy[i] - entropy[j])
        return metric_matrix

    def sample_clients_entropy(self, entropy, Clusters, n_samples,epoch):
        n_sampled = max(int(self.args["frac"] * self.args["n_clients"]), 1)
        n_clients = len(n_samples)
        n_clustered = len(Clusters)
        entropy = np.exp(self.args["gamma"]*(self.args["epochs"]- epoch)*entropy/self.args["epochs"])
    
        p_cluster = entropy/np.sum(entropy)
        sampled_clients = []
        clusters_selected = [0]*n_sampled
        
        for k in range(n_sampled):
            select_group = int(choice(n_clustered, 1, p=p_cluster))
            while clusters_selected[select_group] >= len(Clusters[select_group]):
                select_group = int(choice(n_clustered, 1, p=p_cluster)) 
            clusters_selected[select_group] += 1
        
        for k in range(len(clusters_selected)):
            if clusters_selected[k] == 0:
                continue
            select_clients = choice(Clusters[k], clusters_selected[k], replace = False)
            for i in range(clusters_selected[k]):
                sampled_clients.append(select_clients[i])
        return sampled_clients

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