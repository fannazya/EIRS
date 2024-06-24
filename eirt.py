import math
import torch
from torch import nn
from tqdm import tqdm
from backbones import IRTNet  
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
import numpy as np
import pandas as pd
from longling.lib.structure import AttrDict
import logging
from utils import data_etl, extract_item, information_sample
import pickle
import argparse
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IRT(nn.Module):
    def __init__(self, user_num, item_num, args_config):
        super(IRT, self).__init__()
        self.net = IRTNet(user_num, item_num)
        self.config = args_config

    def get_weights(self, users, infos):
        
        info_num = infos.size(-1)
        bz = infos.size(0)
        net = self.net.to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.config.lr)

        # Freeze parameters of the network except for theta
        for name, param in net.named_parameters():
            if 'theta' not in name:
                param.requires_grad = False

        original_weights = net.theta.weight.data.clone()
        users = users.to(device)
        infos = infos.to(device)
        updates = torch.tensor([]).to(device)
        point_function = nn.BCELoss()
        
        for i in range(info_num):
            info = infos[:, i]  
            correct = torch.tensor([1]* bz, device=device).float()
            wrong = torch.tensor([0]* bz, device=device).float()

            optimizer.zero_grad()
            pred = net(users, info)
            loss_correct = point_function(pred, correct)
            loss_correct.backward()
            optimizer.step()

            up_weights = net.theta.weight.data.clone()
            net.theta.weight.data.copy_(original_weights)

            optimizer.zero_grad()
            pred = net(users, info)
            loss_wrong = point_function(pred, wrong)
            loss_wrong.backward()
            optimizer.step()

            down_weights = net.theta.weight.data.clone()
            net.theta.weight.data.copy_(original_weights)

            update = pred * torch.norm(up_weights - original_weights).item() + \
                     (1 - pred) * torch.norm(down_weights - original_weights).item()
            updates = torch.cat((updates, update.unsqueeze(0)), dim=0)

        weights = F.softmax(updates, dim=0)
        for param in net.parameters():
            param.requires_grad = True
        return weights.transpose(0, 1)

    def train(self, train_data, test_data=None) -> ...:
        epoch=self.config.epoch
        self.net = self.net.to(device)
        trainer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr)
        loss_means = []
        for e in range(epoch): 
            
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):

                # batch data 
                user_id, item_id, _, score = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                self.bz = user_id.size(0)
                score: torch.Tensor = score.to(device)
               
                pred: torch.Tensor = self.net(user_id, item_id)
                pred_score = pred.detach()
                
                # get candidates
                candits = self.net.item_sim_sample(user_id, score, response_pool,
                                                    self.config.n_inter,self.config.n_non) 
                can_num = candits.size(1)
                can_user = user_id.unsqueeze(-1).repeat(1,can_num)
                can_score = self.net(can_user, candits).to(device)
                can_score = can_score.detach()
            
                # informativeness
                info_item = information_sample(pred_score, can_score, candits, self.config.info_num)
                user_id_info = user_id.unsqueeze(-1).repeat(1,self.config.info_num)
                info_pred = self.net(user_id_info, info_item)

        
                weights = self.get_weights(user_id, info_item).detach()
                zero_indices = torch.all(info_item == 0, dim=1)
                weights[zero_indices,:] = 0
              

                # loss
                temp =(e+1)/epoch
                loss = self.net._loss_ours(pred,info_pred,score.float(), weights, self.config.l2, temp)
                trainer.zero_grad()
                loss.backward()
                trainer.step()
 
            
            
            if test_data is not None:
                auc, accuracy, rmse = self.eval(test_data, device=device)
                print("auc: %.6f, accuracy: %.6f, rmse: %.6f " % (auc, accuracy, rmse))
    
 

    def eval(self, test_data, device="cpu") -> tuple:
        self.net = self.net.to(device)
        self.net.eval()
        y_pred = []
        y_true = []
 
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, _, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            response: torch.Tensor = response.to(device)
            pred: torch.Tensor = self.net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5), \
            math.sqrt(mean_squared_error(y_true, y_pred)) 

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bz", type=int, default=256, help="batch_size"
    )
    parser.add_argument(
        "--epoch", type=int, default=25, help="number of training epochs"
    )
    parser.add_argument(
        "--dataset", type=str, default='junyi', help="dataset name"
    )
    parser.add_argument(
        "--output", type=str, default='output', help="output_dir"
    )
    parser.add_argument(
        "--n_inter", type=int, default=5, help="number of interative samples"
    )
    parser.add_argument(
        "--n_non", type=int, default=100, help="number of non-interactive samples"
    )
    parser.add_argument(
        "--info_num", type=int, default=5, help="number of info samples"
    )
    parser.add_argument(
        "--lr", type=float, default=0.002, help="Learning rate for cdm."
    )
    parser.add_argument(
        "--l2", type=float, default=5e-6, help="l2 regularization"
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.INFO)
    
    params = AttrDict(
        hyper_params={"user_num": 10000, "knowledge_num": 39, "item_num": 714},
    )

    train_path ='data/junyi/full_train.csv'
    response_path = 'user_response_pool.pkl'
    f1 = open(response_path, 'rb')
  
    response_pool = pickle.load(f1)

    item_knowledge, knowledge_item = extract_item('data/junyi/items.csv',  params)
    
    train_data,_ = data_etl(train_path, item_knowledge, args)
    valid_data, _ = data_etl('data/junyi/full_valid.csv', item_knowledge, args)
    test_data, _ = data_etl('data/junyi/full_test.csv', item_knowledge, args)
 

    cdm = IRT(
        params['hyper_params']['user_num']+1,
        params['hyper_params']['item_num']+1,
        args_config=args,
    )

    cdm.train(train_data,valid_data)

    auc, accuracy, rmse = cdm.eval(test_data)
    print("auc: %.6f, accuracy: %.6f,rmse: %.6f" % (auc, accuracy, rmse))
