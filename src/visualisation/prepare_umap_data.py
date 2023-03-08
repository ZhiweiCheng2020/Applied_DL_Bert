import torch
import os
import sys
import random
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
curr_path = os.getcwd()
sys.path.append(curr_path)
sys.path.insert(0, os.path.join(curr_path, "src"))
import src.data.data_preprocess as preprocess
from src.models.utils import compute_loss 
import pickle

# load all data
len_all = 13478 # the length of the whole dataset
batch_size = 64
dataset = preprocess.seq_dataset(data_path=os.path.join(curr_path, 
                                            "data/protein_data.csv"), 
                                      seqs_range = random.sample(range(0, len_all), len_all), 
                                      seed=42)
data_loader = DataLoader(dataset=dataset, 
                                        batch_size=batch_size, 
                                        shuffle=True,)

# load saved model
model_path = r"models\saved_model.pth"
Bert_model = torch.load(model_path)
Bert_model.eval()

# a simple function to transform one-hot-encoding to intergers
def one_hot_to_int(one_hot_tensor):
    np_arr = one_hot_tensor.numpy()
    int_version = list(np.argmax(np_arr, axis=1))
    return int_version

# inference, generate embedding
CE_loss = torch.nn.CrossEntropyLoss()
with torch.no_grad():
    for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(data_loader):
        # remove the extra dimension
        token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
        mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
            code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
        MaskedLM, code0_pred, code1_pred, NSP = Bert_model(input=token_data, padding_mask=padding_mask)
        curr_loss, loss_maskLM, loss_code0, loss_code1 = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                            code0, code0_pred,
                            code1, code1_pred,
                            loss_fun=CE_loss)
        
        if i == 0:
            NSP_ebd = NSP
            code0_int = one_hot_to_int(code0)
            code1_int = one_hot_to_int(code1)
        else:
            NSP_ebd = torch.cat((NSP_ebd, NSP), 0)
            code0_int.extend(one_hot_to_int(code0))
            code1_int.extend(one_hot_to_int(code1))

# save embedding for future visulisation
with open(os.path.join(curr_path, "models", "bert_ebd.pkl"), "wb") as f:
    pickle.dump([NSP_ebd, code0_int, code1_int], f)    
    

