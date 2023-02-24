import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
random.seed(42)
np.random.seed(42)
import torch
import sys
import os
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
# from sklearn.model_selection import KFold
torch.random.manual_seed(42)
g = torch.Generator()
g.manual_seed(42)

curr_path = os.getcwd()
print(f"curr_path: {curr_path}")
# sys.path.insert(0, os.path.dirname(curr_path))
import data.data_preprocess as preprocess
import models.bert as bert 


# len_all = 13478 # the length of the whole dataset
len_all = 1000 # the length of the whole dataset

# model parameters
ebd_dim = 60 # Embedding Size
vocab_size = 24
max_seq_len = 182 # maximum sequence length is 181, but we add a [CLS] before it.
num_layer = 2 # number of Encoder of Encoder Layer
num_head = 2 # number of heads in Multi-Head Attention
feedforward_dim = ebd_dim * 4  # 4*ebd_dim, FeedForward dimension

# training parameters
lr=0.0002
num_epochs = 3
batch_size = 64
verbose = True

# split the data to train (0.85*0.8), validation (0.85*0.2), and test (0.15) sets
test_set_size = int(len_all * 0.15)
test_idx = random.sample(range(0, len_all), test_set_size)
train_val_idx = list(set(list(range(len_all))) - set(test_idx))
test_dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path, 
                                            "data/protein_data.csv"), seqs_range = test_idx, seed=42)
train_val_dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path,
                                            "data/protein_data.csv"), seqs_range = train_val_idx, seed=42)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, 
                                            [train_size, val_size],
                                generator=torch.Generator().manual_seed(42),
                                )
train_loader = DataLoader(dataset=train_dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True, generator=g,)
val_loader = DataLoader(dataset=val_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, generator=g,)

# training and validation
train_loss_history = []
val_loss_history = []
min_val_loss = np.inf # to track the minimal validation loss
CE_loss = nn.CrossEntropyLoss()
Bert_model = bert.Bert(ebd_dim=ebd_dim, num_head=num_head, vocab_size=vocab_size,
                        feedforward_dim=feedforward_dim,
                        num_layer=num_layer, max_seq_len=max_seq_len)

for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    optimizer = optim.Adam(Bert_model.parameters(), lr=lr)
    
    # training
    for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(train_loader):
        # remove the extra dimension
        token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
        mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
            code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
        # train the model
        optimizer.zero_grad()
        MaskedLM, code0_pred, code1_pred, code2_pred = Bert_model(input=token_data, padding_mask=padding_mask)
        # masked token prediction
        # (batch_size, n_mask) --> (batch_size,n_mask,vocab_size)
        masked_pos = mask_pos_data.unsqueeze(-1).expand(-1, -1, MaskedLM.size(-1))
        # (batch_size,source_len,vocab_size) --> (batch_size,n_mask,vocab_size)
        MaskedLM = torch.gather(MaskedLM, 1, masked_pos)
        # calculate the training loss       
        loss_maskLM = CE_loss(MaskedLM.transpose(1,2), mask_token_data)
        loss_code0 = CE_loss(code0_pred, code0.float())
        loss_code1 = CE_loss(code1_pred, code1.float())
        # loss_code2 = CE_loss(code2_pred, code2.float())
        # curr_loss = loss_maskLM + loss_code0
        curr_loss = loss_maskLM + loss_code0 + loss_code1
        curr_loss.backward()
        optimizer.step()
        train_loss += curr_loss.item()
        
        
    
    # record train loss
    epoch_train_loss = train_loss/len(train_loader)
    train_loss_history.append(epoch_train_loss)
    
    # validation
    with torch.no_grad():
        for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(val_loader):
            # remove the extra dimension
            token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
            mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
            MaskedLM, code0_pred, code1_pred, code2_pred = Bert_model(input=token_data, padding_mask=padding_mask)
            # masked token prediction
            # (batch_size, n_mask) --> (batch_size,n_mask,vocab_size)
            masked_pos = mask_pos_data.unsqueeze(-1).expand(-1, -1, MaskedLM.size(-1))
            # (batch_size,source_len,vocab_size) --> (batch_size,n_mask,vocab_size)
            MaskedLM = torch.gather(MaskedLM, 1, masked_pos)
            # calculate the val loss       
            loss_maskLM = CE_loss(MaskedLM.transpose(1,2), mask_token_data)
            loss_code0 = CE_loss(code0_pred, code0.float())
            loss_code1 = CE_loss(code1_pred, code1.float())
            # loss_code2 = CE_loss(code2_pred, code2.float())
            # curr_loss = loss_maskLM + loss_code0
            curr_loss = loss_maskLM + loss_code0 + loss_code1
            val_loss += curr_loss.item()
            
            if ((i+1) % 10 == 0) and verbose:
                num_iters = math.ceil(len(val_dataset)/batch_size)
                print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{num_iters}, validation loss,\
                    loss_maskLM: {loss_maskLM.item():.3f}\
                        loss_code0: {loss_code0.item():.3f}\
                            loss_code1: {loss_code1.item():.3f}'
                            )
            
    # record train loss
    epoch_val_loss = val_loss/len(val_loader)
    val_loss_history.append(epoch_val_loss)
    
    if verbose:
        print(f"Epoch {epoch+1}, training loss: {epoch_train_loss:.5f}, validation loss: {epoch_val_loss:.5f}")
    
    if epoch_val_loss < min_val_loss:
        if verbose:
            print(f'Validation loss decreased: ({min_val_loss:.5f}-->{epoch_val_loss:.5f}), saving the model.')
        min_val_loss = epoch_val_loss
        
        # save the model
        torch.save(Bert_model.state_dict(),os.path.join(curr_path, "models", 'saved_model.pth'))

# plot the loss
plt.plot(train_loss_history, color="blue", label = "Training Loss")
plt.plot(val_loss_history, color="red", label = "Validation Loss")
plt.legend(loc="upper right")
plt.xlabel("Epochs")
plt.ylabel("Train/Val Loss")
# plt.show()
plt.savefig(os.path.join(curr_path, "results", "train_val_loss.pdf"), dpi=150)