import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def sample_sim(pool, num_samples):
    if len(pool) >= num_samples:
        return np.random.choice(pool, size=num_samples, replace=False)
    elif len(pool) != 0:
        return np.random.choice(pool, size=num_samples, replace=True).tolist()
    return [0]*num_samples


def irt2pl(theta, a, b, *, F=np):
    """

    Parameters
    ----------
    theta
    a
    b
    F

    Returns
    -------

    Examples
    --------
    >>> theta = [1, 0.5, 0.3]
    >>> a = [-3, 1, 3]
    >>> b = 0.5
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    0.109...
    >>> theta = [[1, 0.5, 0.3], [2, 1, 0]]
    >>> a = [[-3, 1, 3], [-3, 1, 3]]
    >>> b = [0.5, 0.5]
    >>> irt2pl(theta, a, b) # doctest: +ELLIPSIS
    array([0.109..., 0.004...])
    """
    return 1 / (1 + F.exp(- F.sum(F.multiply(a, theta), axis=-1) + b))


class MIRTNet(nn.Module):
    def __init__(self, user_num, item_num, latent_dim, a_range=None, irf_kwargs=None):
        super(MIRTNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.irf_kwargs = irf_kwargs if irf_kwargs is not None else {}
        self.theta = nn.Embedding(self.user_num, latent_dim)
        self.a = nn.Embedding(self.item_num, latent_dim)
        self.b = nn.Embedding(self.item_num, 1)
        


    def item_similarity(self, inter_item, non_item, num_samples):
        # preparation: params and id tensor
        
        inter_item_tensor = torch.tensor(inter_item).unsqueeze(0)
        non_item_tensor = torch.tensor(non_item).unsqueeze(-1)

        # difficulty
        inter_beta_e = self.a(inter_item_tensor) 
        non_beta_e = self.a(non_item_tensor) 
        dif_score = torch.sum(torch.sum(inter_beta_e * non_beta_e, dim=1),dim=-1)
    
     
   
        # discrimination
        inter_alp_e = self.b(inter_item_tensor)
        non_alp_e = self.b(non_item_tensor)
        dis_score = torch.sum(torch.sum(inter_alp_e*non_alp_e, dim=1),dim=-1)
        
        # sum of two embedding score
        sim_score =(F.softmax(dif_score,dim=-1)+F.softmax(dis_score,dim=-1))/2
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
 

    def forward(self, user, item):
        theta = torch.squeeze(self.theta(user), dim=-1)
        a = torch.sigmoid(torch.squeeze(self.a(item), dim=-1))
        b = torch.squeeze(self.b(item), dim=-1)
        if torch.max(theta != theta) or torch.max(a != a) or torch.max(b != b):  # pragma: no cover
            raise ValueError('ValueError:theta,a,b may contains nan!  The a_range is too large.')
        return self.irf(theta, a, b, **self.irf_kwargs)
 
        
         
    @classmethod
    def irf(cls, theta, a, b, **kwargs):
        return irt2pl(theta, a, b, F=torch)
    

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
 

 