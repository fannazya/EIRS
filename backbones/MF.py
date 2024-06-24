import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def sample_sim(pool, num_samples):
    if len(pool) >= num_samples:
        return np.random.choice(pool, size=num_samples, replace=False)
    elif len(pool) != 0:
        return np.random.choice(pool, size=num_samples, replace=True).tolist()
    return [0]*num_samples

class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)

        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)

    def item_similarity(self, inter_item, non_item, num_samples):
        # preparation: params and id tensor
        
        inter_item_tensor = torch.tensor(inter_item).unsqueeze(0)
        non_item_tensor = torch.tensor(non_item).unsqueeze(-1)

        inter_alp_e =  self.item_embedding(inter_item_tensor)
        non_alp_e =  self.item_embedding(non_item_tensor)
        sim_score = torch.sum(torch.sum(inter_alp_e*non_alp_e, dim=1),dim=-1)
        
        sim_score =F.softmax(sim_score,dim=0)
 
        # sample according to the probability
        sampled_index = torch.multinomial(sim_score, num_samples, replacement=True)

        return  non_item_tensor[sampled_index]
    
 

    def item_sim_sample(self, users, scores, item_pool, n_inter, n_non):
        can_samples = []
      
        for user, label in zip(users, scores):
            non_response = 1 - label.item()
            sim_response = non_response + 2
            
            non_pool = item_pool[user.item()][non_response]
            sim_pool = item_pool[user.item()][sim_response]

            sim_items = sample_sim(sim_pool, n_inter)

            # Check if non_pool is empty
            if len(non_pool) != 0:
                can_item = self.item_similarity(sim_items, non_pool, n_non)
            else:
                # If non_pool is empty, create tensor of zeros
                can_item = torch.zeros(n_non, dtype=torch.int64)
        
            can_samples.append(can_item.clone().detach().squeeze())
    
 
        return torch.stack(can_samples) 
    

    def _loss_ours(self, pred1, pred2, score, weights, l2_lambda, temp, *args):
        
        CE_loss= nn.BCELoss()

        score = score.unsqueeze(-1).repeat(1, pred2.size(1))
        pred1 = pred1.unsqueeze(-1).repeat(1, pred2.size(1))
        pairwise_loss = -(torch.log(torch.sigmoid(score*(pred1 - pred2))) + torch.log(torch.sigmoid((score-1)*(pred1 - pred2)))) 

        # Calculate L2 regularization term
        l2_regularization = sum(torch.norm(param)**2 for param in self.parameters())
        partial_loss =  torch.sum(weights*pairwise_loss,dim=-1).mean() 
        loss = CE_loss(pred1,score) + temp*partial_loss+ l2_lambda * l2_regularization
        return loss
 