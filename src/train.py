import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import random
random.seed(42)
np.random.seed(42)
import torch
from torch.optim.lr_scheduler import ExponentialLR
import sys
import os
os.environ['PYTHONHASHSEED']=str(42)
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math
# from sklearn.model_selection import KFold
torch.random.manual_seed(42)
g = torch.Generator()
g.manual_seed(42)
curr_path = os.getcwd()
import data.data_preprocess as preprocess
import models.bert as bert 
from models.utils import compute_loss 
from models.utils import EarlyStopper 

# data set size
len_all = 13478 # the length of the whole dataset
# len_all = 1000 # the length of the whole dataset

# training parameters
lr=0.001 #learning rate
num_epochs = 100
batch_size = 64
verbose = True # if show training process

# model parameters
ebd_dim = 60 # Embedding Size
vocab_size = 24 # number of different letters in sequence
max_seq_len = 182 # maximum sequence length is 181, but we add a [CLS] before it.
num_layer = 4 # number of Encoder Layers
num_head = 4 # number of heads in Multi-Head Attention
feedforward_dim = ebd_dim * 4  # 4*ebd_dim, FeedForward dimension

# split the data to train (0.85*0.8), validation (0.85*0.2), and test (0.15) sets
test_set_size = int(len_all * 0.15)
test_idx = random.sample(range(0, len_all), test_set_size)
train_val_idx = list(set(list(range(len_all))) - set(test_idx))
test_dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path, 
                                            "data/protein_data.csv"), 
                                      seqs_range = test_idx, seed=42)
train_val_dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path,
                                            "data/protein_data.csv"), 
                                           seqs_range = train_val_idx, seed=42)
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
test_loader = DataLoader(dataset=test_dataset, 
                                    batch_size=batch_size, 
                                    shuffle=True, generator=g,)

# model setting
CE_loss = nn.CrossEntropyLoss()
Bert_model = bert.Bert(ebd_dim=ebd_dim, num_head=num_head, vocab_size=vocab_size,
                        feedforward_dim=feedforward_dim,
                        num_layer=num_layer, max_seq_len=max_seq_len)
optimizer = optim.Adam(Bert_model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98, verbose=verbose)
early_stopper = EarlyStopper(patience=3, min_delta=0.0001)

# training and validation
train_loss_history = []
val_loss_history = []
test_loss_history = []
test_MLM_loss_history = []
test_code0_loss_history = []
test_code1_loss_history = []
min_val_loss = np.inf # to track the minimal validation loss

# start
for epoch in range(num_epochs):
    train_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    test_MLM_loss = 0.0
    test_code0_loss = 0.0
    test_code1_loss = 0.0
    
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
        curr_loss,_,_,_ = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                 code0, code0_pred,
                 code1, code1_pred,
                 loss_fun=CE_loss)
        curr_loss.backward()
        optimizer.step()
        train_loss += curr_loss.item()
    # record train loss
    epoch_train_loss = train_loss/len(train_loader)
    train_loss_history.append(epoch_train_loss)
    
    # start reducing learning rate exponentially after some epochs
    if epoch > 10:
        scheduler.step()
        
    # get learning rate
    my_lr = scheduler.get_lr()[0]
    print(f'The current lr is: {my_lr:.5f}')
    # validation
    with torch.no_grad():
        for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(val_loader):
            # remove the extra dimension
            token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
            mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
            MaskedLM, code0_pred, code1_pred, code2_pred = Bert_model(input=token_data, padding_mask=padding_mask)
            # masked token prediction
            curr_loss,_,_,_ = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                                code0, code0_pred,
                                code1, code1_pred,
                                loss_fun=CE_loss)
            val_loss += curr_loss.item()
            
    # record train loss
    epoch_val_loss = val_loss/len(val_loader)
    val_loss_history.append(epoch_val_loss)
    if early_stopper.early_stop(epoch_val_loss):             
        break
    
    if epoch_val_loss < min_val_loss:
        if verbose:
            print(f'Validation loss decreased: ({min_val_loss:.5f}-->{epoch_val_loss:.5f}), saving the model.')
        min_val_loss = epoch_val_loss
        
        # save the model
        torch.save(Bert_model,os.path.join(curr_path, "models", 'saved_model.pth'))

    # test
    with torch.no_grad():
        for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(test_loader):
            # remove the extra dimension
            token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
            mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
            MaskedLM, code0_pred, code1_pred, code2_pred = Bert_model(input=token_data, padding_mask=padding_mask)
            # masked token prediction
            curr_loss, loss_maskLM, loss_code0, loss_code1 = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                                code0, code0_pred,
                                code1, code1_pred,
                                loss_fun=CE_loss)
            test_loss += curr_loss.item()
            test_MLM_loss += loss_maskLM.item()
            test_code0_loss += loss_code0.item()
            test_code1_loss += loss_code1.item()
            
            # if ((i+1) % 10 == 0) and verbose:
            #     num_iters = math.ceil(len(test_dataset)/batch_size)
            #     print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{num_iters}, test loss,\
            #         loss_maskLM: {loss_maskLM.item():.3f}\
            #             loss_code0: {loss_code0.item():.3f}\
            #                 loss_code1: {loss_code1.item():.3f}'
            #                 )
            
    # record test loss
    epoch_test_loss = test_loss/len(test_loader)
    test_loss_history.append(epoch_test_loss)
    test_MLM_loss_history.append(test_MLM_loss/len(test_loader))
    test_code0_loss_history.append(test_code0_loss/len(test_loader))
    test_code1_loss_history.append(test_code1_loss/len(test_loader))
    
    if verbose:
        print(f"Epoch {epoch+1}, training loss: {epoch_train_loss:.5f}, validation loss: {epoch_val_loss:.5f}, test loss: {epoch_test_loss:.5f}")

     
# plot the Train / Val / Test Los
plt.plot(train_loss_history, color="blue", label = "Training Loss")
plt.plot(val_loss_history, color="red", label = "Validation Loss")
plt.plot(test_loss_history, color="green", label = "Test Loss")
plt.legend(loc="upper right")
plt.xlabel("Epochs")
plt.ylabel("Train / Val / Test Loss")
plt.title("Train / Val / Test Loss")
plt.savefig(os.path.join(curr_path, "results", 
        "TrainValTestloss_"+str(len_all)+"_seqs_"+str(num_epochs)+"_epochs_"+str(num_head)+\
            "_heads_"+str(num_layer)+"_layers.pdf"), dpi=150)

# clear the plot
plt.clf()

# plot the Train / Val / Test Los
plt.plot(test_MLM_loss_history, color="blue", label = "MLM Loss")
plt.plot(test_code0_loss_history, color="red", label = "code0 Loss")
plt.plot(test_code1_loss_history, color="green", label = "code1 Loss")
plt.legend(loc="upper right")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
plt.title("Test Loss Decomposition")
plt.savefig(os.path.join(curr_path, "results", 
        "Testloss_"+str(len_all)+"_seqs_"+str(num_epochs)+"_epochs_"+str(num_head)+\
            "_heads_"+str(num_layer)+"_layers.pdf"), dpi=150)