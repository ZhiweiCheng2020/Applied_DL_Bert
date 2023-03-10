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

import data.data_preprocess as preprocess
import models.bert as bert 
from models.utils import compute_loss 
from models.utils import EarlyStopper 

class train:
    def __init__(self, 
                len_all = 13478, # the whole dataset
                lr=0.001, #learning rate
                num_epochs = 100,
                batch_size = 64,
                verbose = True, # if show training process
                ebd_dim = 60, # Embedding Size
                vocab_size = 24, # number of different letters in sequence
                max_seq_len = 182, # maximum sequence length is 181, but we add a [CLS] before it.
                num_layer = 4, # number of Encoder Layers
                num_head = 4, # number of heads in Multi-Head Attention
        ):
        self.len_all = len_all
        self.lr=lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.verbose = verbose # if show training process
        self.ebd_dim = ebd_dim # Embedding Size
        self.vocab_size = vocab_size # number of different letters in sequence
        self.max_seq_len = max_seq_len # maximum sequence length is 181, but we add a [CLS] before it.
        self.num_layer = num_layer # number of Encoder Layers
        self.num_head = num_head # number of heads in Multi-Head Attention):
        self.feedforward_dim = ebd_dim * 4  # 4*ebd_dim, FeedForward dimension
        self.curr_path = os.path.dirname(os.getcwd())
    
    def run_model(self):
        self.create_dataloader()
        self.model_setup()
        self.train_val()
        self.plot_loss()
        return self.model_path
        
    def create_dataloader(self):
        # split the data to train (0.85*0.8), validation (0.85*0.2), and test (0.15) sets
        all_idx = random.sample(range(0, 13478), self.len_all)
        test_set_size = int(self.len_all * 0.15)
        test_idx = random.sample(all_idx, test_set_size)
        train_val_idx = list(set(list(all_idx)) - set(test_idx))
        test_dataset = preprocess.seq_dataset(data_path=os.path.join(self.curr_path, 
                                                    "data", "protein_data.csv"), 
                                            seqs_range = test_idx, seed=42)
        train_val_dataset = preprocess.seq_dataset(data_path=os.path.join(self.curr_path,
                                                    "data/protein_data.csv"), 
                                                seqs_range = train_val_idx, seed=42)
        train_size = int(0.8 * len(train_val_dataset))
        val_size = len(train_val_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, 
                                                    [train_size, val_size],
                                        generator=torch.Generator().manual_seed(42),
                                        )
        self.train_loader = DataLoader(dataset=train_dataset, 
                                                batch_size=self.batch_size, 
                                                shuffle=True, generator=g,)
        self.val_loader = DataLoader(dataset=val_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True, generator=g,)
        self.test_loader = DataLoader(dataset=test_dataset, 
                                            batch_size=self.batch_size, 
                                            shuffle=True, generator=g,)

    def model_setup(self):
        # model setting
        self.CE_loss = nn.CrossEntropyLoss()
        self.Bert_model = bert.Bert(ebd_dim=self.ebd_dim, num_head=self.num_head, vocab_size=self.vocab_size,
                                feedforward_dim=self.feedforward_dim,
                                num_layer=self.num_layer, max_seq_len=self.max_seq_len)
        self.optimizer = optim.Adam(self.Bert_model.parameters(), lr=self.lr)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=0.98, verbose=self.verbose)
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.0001)
        # training and validation
        self.train_loss_history = []
        self.val_loss_history = []
        self.test_loss_history = []
        self.test_MLM_loss_history = []
        self.test_code0_loss_history = []
        self.test_code1_loss_history = []
        self.min_val_loss = np.inf # to track the minimal validation loss

    def train_val(self):
        # start
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            val_loss = 0.0
            test_loss = 0.0
            test_MLM_loss = 0.0
            test_code0_loss = 0.0
            test_code1_loss = 0.0
            
            # training
            for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(self.train_loader):
                # remove the extra dimension
                token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
                mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                    code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
                # train the model
                self.optimizer.zero_grad()
                MaskedLM, code0_pred, code1_pred, code2_pred, _ = self.Bert_model(input=token_data, padding_mask=padding_mask)
                # masked token prediction
                curr_loss,_,_,_ = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                        code0, code0_pred,
                        code1, code1_pred,
                        loss_fun=self.CE_loss)
                curr_loss.backward()
                self.optimizer.step()
                train_loss += curr_loss.item()
            # record train loss
            epoch_train_loss = train_loss/len(self.train_loader)
            self.train_loss_history.append(epoch_train_loss)
            
            # start reducing learning rate exponentially after some epochs
            if epoch > 10:
                self.scheduler.step()
                
            # get learning rate
            my_lr = self.scheduler.get_lr()[0]
            print(f'The current lr is: {my_lr:.5f}')
            # validation
            with torch.no_grad():
                for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(self.val_loader):
                    # remove the extra dimension
                    token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
                    mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                        code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
                    MaskedLM, code0_pred, code1_pred, code2_pred, _ = self.Bert_model(input=token_data, padding_mask=padding_mask)
                    # masked token prediction
                    curr_loss,_,_,_ = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                                        code0, code0_pred,
                                        code1, code1_pred,
                                        loss_fun=self.CE_loss)
                    val_loss += curr_loss.item()
                    
            # record train loss
            epoch_val_loss = val_loss/len(self.val_loader)
            self.val_loss_history.append(epoch_val_loss)
            if self.early_stopper.early_stop(epoch_val_loss):             
                break
            
            if epoch_val_loss < self.min_val_loss:
                if self.verbose:
                    print(f'Validation loss decreased: ({self.min_val_loss:.5f}-->{epoch_val_loss:.5f}), saving the model.')
                self.min_val_loss = epoch_val_loss
                self.model_path = os.path.join(self.curr_path, "models", 'saved_model.pth')
                # save the model
                torch.save(self.Bert_model, self.model_path)

            # test
            with torch.no_grad():
                for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(self.test_loader):
                    # remove the extra dimension
                    token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
                    mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                        code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
                    MaskedLM, code0_pred, code1_pred, code2_pred, _ = self.Bert_model(input=token_data, padding_mask=padding_mask)
                    # masked token prediction
                    curr_loss, loss_maskLM, loss_code0, loss_code1 = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                                        code0, code0_pred,
                                        code1, code1_pred,
                                        loss_fun=self.CE_loss)
                    test_loss += curr_loss.item()
                    test_MLM_loss += loss_maskLM.item()
                    test_code0_loss += loss_code0.item()
                    test_code1_loss += loss_code1.item()
                    
            # record test loss
            epoch_test_loss = test_loss/len(self.test_loader)
            self.test_loss_history.append(epoch_test_loss)
            self.test_MLM_loss_history.append(test_MLM_loss/len(self.test_loader))
            self.test_code0_loss_history.append(test_code0_loss/len(self.test_loader))
            self.test_code1_loss_history.append(test_code1_loss/len(self.test_loader))
            
            if self.verbose:
                print(f"Epoch {epoch+1}, training loss: {epoch_train_loss:.5f}, validation loss: {epoch_val_loss:.5f}, test loss: {epoch_test_loss:.5f}")
    
    def plot_loss(self):
        # plot the Train / Val / Test Los
        plt.plot(self.train_loss_history, color="blue", label = "Training Loss")
        plt.plot(self.val_loss_history, color="red", label = "Validation Loss")
        plt.plot(self.test_loss_history, color="green", label = "Test Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epochs")
        plt.ylabel("Train / Val / Test Loss")
        plt.title("Train / Val / Test Loss")
        plt.savefig(os.path.join(self.curr_path, "results", 
                "TrainValTestloss_"+str(self.len_all)+"_seqs_"+str(self.num_epochs)+"_epochs_"+str(self.num_head)+\
                    "_heads_"+str(self.num_layer)+"_layers.pdf"), dpi=150)
        # clear the plot
        plt.clf()

        # plot the Train / Val / Test Los
        plt.plot(self.test_MLM_loss_history, color="blue", label = "MLM Loss")
        plt.plot(self.test_code0_loss_history, color="red", label = "code0 Loss")
        plt.plot(self.test_code1_loss_history, color="green", label = "code1 Loss")
        plt.legend(loc="upper right")
        plt.xlabel("Epochs")
        plt.ylabel(" Loss")
        plt.title("Test Loss Decomposition")
        plt.savefig(os.path.join(self.curr_path, "results", 
                "Testloss_"+str(self.len_all)+"_seqs_"+str(self.num_epochs)+"_epochs_"+str(self.num_head)+\
                    "_heads_"+str(self.num_layer)+"_layers.pdf"), dpi=150)
        

if __name__ == "__main__":
    train().run_model()