import torch


def information_sample(score,can_score,candits,info_num,can_kcs=None):
    can_num = candits.size(1)
    bz = candits.size(0)
    score = score.unsqueeze(1).repeat(1, can_num)
    inf = abs(score-can_score)
    _, indices = torch.sort(inf, descending=False)
    nid = indices[:,:info_num]
    row_id = torch.arange(bz).unsqueeze(1)
    if can_kcs is not None:
        return candits[row_id, nid], can_kcs[row_id, nid]

    return candits[row_id, nid] 