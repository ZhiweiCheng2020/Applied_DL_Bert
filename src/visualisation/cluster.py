import torch
import os
import sys
import random
import umap
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
curr_path = os.getcwd()
# curr_path = r"C:/Study_courses/2022_winter_LMU/Applied_DL_Bert"
# print(f"curr_path: {curr_path}")
sys.path.append(curr_path)
import src.data.data_preprocess as preprocess
import src.models.bert as bert 
import pickle


# load all data, loader
len_all = 13478 # the length of the whole dataset
batch_size = 64
dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path, 
                                            "data/protein_data.csv"), 
                                      seqs_range = random.sample(range(0, len_all), len_all), 
                                      seed=42)
data_loader = DataLoader(dataset=dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True,)
# load the saved model 
# model parameters
ebd_dim = 60 # Embedding Size
vocab_size = 24 # number of different letters in sequence
max_seq_len = 182 # maximum sequence length is 181, but we add a [CLS] before it.
num_layer = 4 # number of Encoder Layers
num_head = 4 # number of heads in Multi-Head Attention
feedforward_dim = ebd_dim * 4  # 4*ebd_dim, FeedForward dimension

Bert_model = bert.Bert(ebd_dim=ebd_dim, num_head=num_head, vocab_size=vocab_size,
                        feedforward_dim=feedforward_dim,
                        num_layer=num_layer, max_seq_len=max_seq_len)
model_path = r"models\saved_model.pth"
Bert_model.load_state_dict(torch.load(model_path))

def one_hot_to_int(one_hot_tensor):
    np_arr = one_hot_tensor.numpy()
    int_version = list(np.argmax(np_arr, axis=1))
    return int_version

    
with torch.no_grad():
        for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(data_loader):
            # remove the extra dimension
            token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
            mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
            _, code0_pred, code1_pred, _ = Bert_model(input=token_data, padding_mask=padding_mask)
            
            if i == 0:
                code0_ebd = code0_pred
                code1_ebd = code1_pred
                code0_int = one_hot_to_int(code0)
                code1_int = one_hot_to_int(code1)
            else:
                code0_ebd = torch.cat((code0_ebd, code0_pred), 0)
                code1_ebd = torch.cat((code1_ebd, code1_pred), 0)
                code0_int.extend(one_hot_to_int(code0))
                code1_int.extend(one_hot_to_int(code1))

print(code0_ebd.shape)     
print(code1_ebd.shape) 

with open(os.path.join(curr_path, "results", "code0_ebd.pkl"), "wb") as f:
    pickle.dump([code0_ebd, code0_int], f)    
with open(os.path.join(curr_path, "results", "code1_ebd.pkl"), "wb") as f:
    pickle.dump([code1_ebd, code1_int], f)

# umap
embedding = umap.UMAP(n_components=2, min_dist=0.1, 
                      random_state=42, 
                      verbose=True).fit_transform(code0_ebd)

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    embedding[:, 0], embedding[:, 1], s=0.1,
    c=code0_int,
)
plt.title("Code0 embedding projected by UMAP", fontsize=18)

plt.show()