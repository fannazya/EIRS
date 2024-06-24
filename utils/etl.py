import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
import logging
from longling import print_time 
from functools import partial
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
 
def extract_item(data_src,  params,item_map=None):
    with print_time("loading data from %s" % os.path.abspath(data_src), logging.getLogger()):
        data_log = pd.read_csv(data_src)
        knowledge = {}
        knowledge_item = {} 
        for record in tqdm(data_log.to_dict("records"), "reading records from %s" % data_src):
            knowledge_code_vector = [0] * params["hyper_params"]["knowledge_num"]
            for code in eval(record["knowledge_code"]):
                if code >= 1:
                    knowledge_code_vector[code - 1] = 1
            if item_map:
                knowledge[item_map[record["item_id"]]] = np.asarray(knowledge_code_vector)
            else:
                knowledge[record["item_id"]] = np.asarray(knowledge_code_vector)
        return knowledge, knowledge_item 

def extract(data_src):
    with print_time("loading data from %s" % os.path.abspath(data_src), logging.getLogger()):
        df = pd.read_csv(data_src, dtype={"user_id": "int64", "item_id": "int64", "score": "float32"})
        return shuffle(df)


def transform(df, knowledge, *args):
    
    dataset = TensorDataset(
        torch.tensor(df["user_id"], dtype=torch.long),
        torch.tensor(df["item_id"], dtype=torch.long),
        torch.tensor(np.stack([knowledge[int(item)] for item in df["item_id"]])),
        torch.tensor(df["score"], dtype=torch.float)
    )
    return dataset


def load(transformed_data, params):
    batch_size = params.bz

    return DataLoader(transformed_data, batch_size=batch_size)


def data_etl(filepath, knowledge, args):
    raw_data = extract(filepath)
    transformed_data = transform(raw_data, knowledge, args)
    return load(transformed_data, args), raw_data
