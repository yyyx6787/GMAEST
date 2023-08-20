import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings

warnings.filterwarnings('ignore')

import pickle
import dill
import networkx as nx
import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.data import Batch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GNN
from sklearn.metrics import roc_auc_score
import pandas as pd
from copy import deepcopy
from torch_geometric.nn import global_mean_pool, global_add_pool
from loader_aug import Dataset_graphcl
from loader import BioDataset_graphcl1
import random
import copy
k = int(0.7*1388)
percent = 0.7
def load_data(file):
        data_load_file = []
        file_1 = open(file, "rb")
        data_load_file = pickle.load(file_1)
        return data_load_file
    
    
class graphcl(nn.Module):
    def __init__(self, gnn, emb_dim):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, emb_dim))
        self.projection = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True), nn.Linear(emb_dim, 1))
        self.mask_emb = nn.Parameter(torch.Tensor(torch.randn(emb_dim)), requires_grad=True)
        self.mask_type = nn.Parameter(torch.Tensor(torch.randn(1)), requires_grad=True)
        self.mask_attr = nn.Parameter(torch.Tensor(torch.randn(4)), requires_grad=True)
        # self.mask_edge = nn.Parameter(torch.Tensor(torch.randn(2), dtype=torch.float64), requires_grad=True)
        
        

    def forward_cl(self, x, edge_index, edge_attr, edge_type, batch):
        scores_before = torch.squeeze(torch.sigmoid(self.projection(x)), axis=1)
        _, idx1 = torch.sort(scores_before, descending=True)#descending为alse，升序，为True，降序
        idx = idx1[:k]
        # mask node feature
        x[idx] = self.mask_emb
        
        h = self.gnn(x, edge_index.long(),edge_attr.long(), edge_type.long())  #1388 X 96
        # mask corresdoning hidden
        # h[idx] = self.mask_emb
        # mask structure
        # nindex = []
        # ex = edge_index[0,:].cpu().numpy()
        # for sub in idx.cpu().tolist():
        #     indexsub = np.where(ex == sub)[0].tolist()
        #     nindex.extend(indexsub)
        nindex = random.sample([i for i in range(edge_index.size()[1])], int(edge_index.size()[1]*percent))
        # print("nindex:", nindex[:10], len(nindex), nindex1[:10], len(nindex1))
        # println()
        
        edge_index = edge_index.view(2, -1)
        edge_attr[nindex,:] = self.mask_attr
        edge_type[nindex,:] = self.mask_type
        # decoder
        hnew = self.gnn(h, edge_index.long(), edge_attr.long(), edge_type.long())
        # print("fnew:", hnew.size())
        
        
        # x = self.pool(x, batch)
        # print("x:", x.size())
        # x = self.projection_head(x)
        # print("x:", x.size())
        # println()
        return hnew, h, idx, edge_attr, edge_type, nindex
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / 0.5)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        # return -torch.log(
        #     between_sim.diag()
        #     / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())+ 1e-6)
        return -torch.log(
            between_sim.sum(1)
            / (refl_sim.sum(1) + between_sim.sum(1)+ 1e-6))

    def loss_cl_1(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0):
        # h1 = self.projection(z1)
        # h2 = self.projection(z2)
        # print("z1:", z1.size())
        # print("z2:", z2.size())
        simi = torch.exp(nn.CosineSimilarity()(z1,z2)/0.5)
        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)
        ret = (l1 + l2) * 0.5
        # print("simi:", simi)
        # print("l1:", l1)
        # print("l2:", l2)
        # print("ret:", ret)
        # println()
        return ret
    

def custom_collate(data_list):
    # print("used")
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    # batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)    
        
        
def train(args, loader, model_cl, optimizer_cl, device):
    pretrain_loss, generative_loss = 0, 0
    link_loss = 0
    step0 = 0
    for step, batch in enumerate(loader):
        batch, batch1 = batch
        batch, batch1 = batch.to(device), batch1.to(device)      
        # print("---:", batch, batch1)
        
        optimizer_cl.zero_grad()
        edge_attro = copy.deepcopy(batch1.edge_attr)
        edge_typeo = copy.deepcopy(batch1.edge_type)
        
        hnew, h, idx, edge_attr, edge_type, nindex = model_cl.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.edge_type, batch1.batch) # obtain the node feature
        
        # x1 is reconstructed node feature
        # print("**:", edge_attr[nindex][:3], edge_attro[nindex][:3])
        # loss1 = nn.MSELoss()(hnew[idx], h[idx]) # mask node error
        # loss2 = nn.MSELoss()(edge_attr[nindex], torch.nn.functional.normalize(edge_attro[nindex], p=1.0, dim=1))
        # link_loss = nn.MSELoss()(edge_type[nindex], edge_typeo[nindex])
        # loss1 = nn.MSELoss()(hnew, h) # mask node error
        # loss2 = nn.CrossEntropyLoss()(edge_attr, nn.Softmax(dim=-1)(edge_attro))
        # link_loss = nn.MSELoss()(edge_type, edge_typeo)
        # cosine error
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        loss1 = torch.mean(cos(hnew, h))
        loss2 = nn.CrossEntropyLoss()(edge_attr, nn.Softmax(dim=-1)(edge_attro))
        link_loss = nn.MSELoss()(edge_type, edge_typeo)
        # loss2 = nn.MSELoss()(edge_attr[nindex],edge_attro[nindex])
        # link_loss = nn.CrossEntropyLoss()(edge_type, nn.Softmax(dim=0)(edge_typeo))
        # print("--:", edge_attr[:4],edge_attro[:4])
        # println()
        loss = loss1 + loss2 + link_loss
        # print("loss:",  loss1,loss2, link_loss)
        # println()
        
        loss.backward()
        optimizer_cl.step()

        step0 = step
        
       
        file=open(r"./data/hy_vector_gmae_{}_type.pickle".format(k),"wb")
        pickle.dump(h,file) #storing_list
        file.close()
        # loss.backward()
        
        # optimizer_1.step()
        # # optimizer_2.step()
        # generative_loss += float(loss.item())

        # link_loss += (link_loss_1+link_loss_2)/2
        # step0 = step

    return loss/(step+1), loss2/(step+1), link_loss/(step+1)
        
        
        
        
        
def main():
     # Training settings
     parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
     parser.add_argument('--device', type=int, default=0,
                         help='which gpu to use if any (default: 0)')
     parser.add_argument('--batch_size', type=int, default=256,
                         help='input batch size for training (default: 256)')
     parser.add_argument('--epochs', type=int, default=500,
                         help='number of epochs to train (default: 100)')
     parser.add_argument('--lr', type=float, default=0.0005,
                         help='learning rate (default: 0.001)')
     parser.add_argument('--decay', type=float, default=0.01,
                         help='weight decay (default: 0)')
     parser.add_argument('--num_layer', type=int, default=2,
                         help='number of GNN message passing layers (default: 5).')
     parser.add_argument('--emb_dim', type=int, default=96,
                         help='embedding dimensions (default: 300)')
     parser.add_argument('--dropout_ratio', type=float, default=0,
                         help='dropout ratio (default: 0)')
     parser.add_argument('--JK', type=str, default="last",
                         help='how the node features across layers are combined. last, sum, max or concat')
     parser.add_argument('--gnn_type', type=str, default="gcn")
     parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
     parser.add_argument('--aug_mode', type=str, default = 'uniform') 
     parser.add_argument('--aug_strength', type=float, default = 0.5)
     parser.add_argument('--resume', type=int, default=0)
     parser.add_argument('--gamma', type=float, default=0.2)
     args = parser.parse_args()
     device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

     # set up dataset
     # flow_graph = load_data("../data/flow_graph_11.pickle")
     # flow_data = nx_to_graph_data_obj(flow_graph)
     #
     #
     # dataset = (flow_data, flow_data.to_dict())
     # torch.save(dataset, '../data/dataset/processed/dataset.pt')
     root_unsupervised = './data/dataset/'

     dataset = BioDataset_graphcl1(root_unsupervised, data_type='unsupervised')

     dataset.set_augMode(args.aug_mode)
     dataset.set_augStrength(args.aug_strength)
     
     
     gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
     model = graphcl(gnn, args.emb_dim)
     model.to(device)
     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
     # set up vgae model 1
     # gnn_generative_1 = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio,
     #                        gnn_type=args.gnn_type)
     # model_generative_1 = vgae(gnn_generative_1, args.emb_dim)
     # # gnn_generative_1.to(device)
     # optimizer_generative_1 = optim.Adam(model_generative_1.parameters(), lr=args.lr, weight_decay=args.decay)
     
     loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)       
     for epoch in range(args.resume + 1, args.epochs + 1):
         #if epoch == args.epochs:
          #   flag = True
         # loader.dataset.set_generator(deepcopy(model_generative_1).cpu(), deepcopy(model_generative_1).cpu())
         pretrain_loss, generative_loss, link_loss = train(args, loader, model, optimizer, device)
         # pretrain_loss, generative_loss, link_loss = train(args, loader, model_generative_1, device)
         
         # print("pretrain_loss:",pretrain_loss)
         # print("generative_loss:",generative_loss)
         # print()
         if epoch % 10 == 0:
             print("final_loss_train", epoch, pretrain_loss, generative_loss, link_loss)
     # train(args, loader,  device)
        
if __name__ == '__main__': 
    import time
    start = time.time()
    main()        
    print("----cost time----:", time.time()-start)
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    