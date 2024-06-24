import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)

 
def sample_sim(pool, num_samples):
    if len(pool) >= num_samples:
        return np.random.choice(pool, size=num_samples, replace=False)
    elif len(pool) != 0:
        return np.random.choice(pool, size=num_samples, replace=True).tolist()
    return [0]*num_samples
    
class NCDNet(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(NCDNet, self).__init__()

       # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
 
        output_1 = torch.sigmoid(self.prednet_full2(input_x))

        return output_1.squeeze()
 
    
    
    def item_similarity(self, inter_item, non_item, num_samples):
        # preparation: params and id tensor
        
        inter_item_tensor = torch.tensor(inter_item).unsqueeze(0)
        non_item_tensor = torch.tensor(non_item).unsqueeze(-1)

        # difficulty
        inter_beta_e = self.k_difficulty(inter_item_tensor) 
        non_beta_e = self.k_difficulty(non_item_tensor) 
        dif_score = torch.sum(torch.sum(inter_beta_e * non_beta_e, dim=1),dim=-1)
    
     
   
        # discrimination
        inter_alp_e = self.e_difficulty(inter_item_tensor)
        non_alp_e = self.e_difficulty(non_item_tensor)
        dis_score = torch.sum(torch.sum(inter_alp_e*non_alp_e, dim=1),dim=-1)
        
        # sum of two embedding score
        sim_score =(F.softmax(dif_score,dim=-1)+F.softmax(dis_score,dim=-1))/2
        # sample according to the probability
        sampled_index = torch.multinomial(sim_score, num_samples, replacement=True)

        return  non_item_tensor[sampled_index]
    
 

    def item_sim_sample(self, users, scores, item_pool, item_knowledge, n_inter, n_non):
        can_samples = []
        can_kcs = []

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
            
            can_kc = [item_knowledge[item.item()] for item in can_item]
                        
            can_samples.append(can_item.clone().detach().squeeze())
            can_kcs.append(np.array(can_kc)) 

        can_kcs_tensor = torch.tensor(np.array(can_kcs))
        return torch.stack(can_samples), can_kcs_tensor
    
    
    def _loss_ours(self, pred1, pred2, score, weights, l2_lambda, temp, *args):
        
        CE_loss= nn.BCELoss()

        score = score.unsqueeze(-1).repeat(1, pred2.size(1))
        pred1 = pred1.unsqueeze(-1).repeat(1, pred2.size(1))
        pairwise_loss = -(torch.log(torch.sigmoid(score*(pred1 - pred2))) + torch.log(torch.sigmoid((score-1)*(pred1 - pred2)))) 

        # Calculate L2 regularization term
        l2_regularization = sum(torch.norm(param)**2 for param in self.parameters())
        partial_loss =  torch.sum(weights*pairwise_loss,dim=-1).mean() 
        loss = CE_loss(pred1,score) + temp*partial_loss + l2_lambda * l2_regularization
        return loss
 