import numpy as np
import pandas as pd
import random
import torch
import sys
import os
os.environ['PYTHONHASHSEED']=str(42)
from torch.utils.data import Dataset
torch.random.manual_seed(42)
np.random.seed(42)
g = torch.Generator()
g.manual_seed(42)
import copy
import re
import torch.nn.functional as F
# curr_path = os.getcwd()
# sys.path.insert(0, os.path.dirname(curr_path))

# create our dataset
class TokenTransform:
    def __init__(self) -> None:
        pass
    
    # mask the sequence where the token is [PAD]
    
    def create_padding_mask(self, token_data):
        pad_token = self.seq_dict["[PAD]"]
        mask = token_data.data.eq(pad_token) # True if it is [PAD], else false
        return mask
    
    def mask_seqs(self):
        num_unique_tokens = len(self.seq_dict)
        seq_data = [] # save seq info
        
        # deal with the situation when there is only one sequence
        if not isinstance(self.seqs_token[0], list):
            self.seqs_token = [self.seqs_token]

        for token in self.seqs_token:
            
            # get the position of the token in sequence
            token_postion = [idx for idx, val in enumerate(token)
                            if val != self.seq_dict['[CLS]'] and val != self.seq_dict['[SEP]']]
            random.shuffle(token_postion)
            
            mask_position = [] # the position/index of the masked ones in the sequence
            mask_token = [] # the token of the masked ones in the sequence
            n_mask = int(self.max_seq_len * 0.15) #determine the number of masked positions
            
            for i in range(n_mask):
                mask_position.append(token_postion[i]) # add pos
                mask_token.append(token[token_postion[i]]) # add token
                
                if random.random() < 0.8: # 80%: replace by the mask token
                    token[token_postion[i]] = self.seq_dict["[MASK]"]
                elif random.random() < 0.5: # 20%*50% = 10%, replace by a random token
                    # randomly choose a token
                    token[token_postion[i]] = random.randint(0, num_unique_tokens-1)
                # 10%: keep it unchanged.
                
            # add CLS in front of each sequence, which will be used for protein type prediction
            token = [self.seq_dict['[CLS]']] + token
                
            # add padding to make the len(seq) all the same
            zero_pad = self.max_seq_len + 1 - len(token)
            token.extend([self.seq_dict["[PAD]"]] * zero_pad) # seq_dict["[PAD]"] = 0
            seq_data.append([token, mask_position, mask_token,])

        # data for training
        token_data, mask_pos_data, mask_token_data = map(torch.LongTensor, zip(*seq_data))
        
        code0_tmp = torch.tensor(self.code0).to(torch.int64)
        code0 = F.one_hot(code0_tmp, num_classes=3)
        code1_tmp = torch.tensor(self.code1).to(torch.int64)
        code1 = F.one_hot(code1_tmp, num_classes=64) #63+1
        code2_tmp = torch.tensor(self.code2).to(torch.int64)
        code2 = F.one_hot(code2_tmp, num_classes=1027) #1026+1
        # code2 = F.one_hot(code2_tmp, num_classes=1027) #1026+1
        
        pad_mask = self.create_padding_mask(token_data)
        # print(token_data.shape, mask_pos_data.shape, mask_token_data.shape, pad_mask.shape)
        return token_data, mask_pos_data, mask_token_data, code0, code1, code2, pad_mask
        
    def __call__(self, seqs_token, seq_dict, max_seq_len, code0, code1, code2):
        self.seqs_token = seqs_token
        self.seq_dict = seq_dict
        self.max_seq_len = max_seq_len
        if not isinstance(code0, list):
            self.code0 = [code0]
            self.code1 = [code1]
            self.code2 = [code2]
        else:
            self.code0 = code0
            self.code1 = code1
            self.code2 = code2
        return self.mask_seqs()
        

class seq_dataset(Dataset):
    def __init__(self, data_path, 
                 seqs_range,
                 seed = 42,
                 transform = TokenTransform(),):
        self.seed = seed # the seed for random.shuffle
        random.seed(self.seed)
        df = pd.read_csv(data_path, index_col=0)
        df = df.iloc[seqs_range, :]
        df.reset_index(inplace=True)
        self.code0 = df["code0"].to_list()    
        self.code1 = df["code1"].to_list()    
        self.code2 = df["code2"].to_list()    
        self.seqs = df["domain_alpha1_2_sequence"].to_list()        
        self.n_samples = len(self.seqs)
        self.transform = transform
        self.tokenize_seqs()
    
    def tokenize_seqs(self):
        # concate all seqs to one string 
        seq_list = list(set(list("".join(self.seqs))))
        max_seq_len = 0

        # seq to token
        seq_dict = {'[PAD]': 0, '[MASK]': 1, 
                    '[CLS]': 2, '[SEP]': 3, }
        
        init_len = len(seq_dict)
        for idx, val in enumerate(seq_list):
            seq_dict[val] = idx + init_len
        
        # Tokenize the seqs
        seqs_token = list()
        for seq in self.seqs:
            max_seq_len = max(max_seq_len, len(seq)) # get the length of the longest sequence
            arr = [seq_dict[s] for s in list(seq)]
            seqs_token.append(arr)
        
        self.seqs_token = seqs_token
        self.seq_dict = seq_dict
        self.max_seq_len = max_seq_len
        
        # return seqs_token, seq_dict, max_seq_len

    def __getitem__(self, index):
        seq_sample = copy.deepcopy(self.seqs_token[index])
        code0 = copy.deepcopy(self.code0[index])
        code1 = copy.deepcopy(self.code1[index])
        code2 = copy.deepcopy(self.code2[index])
        mask_seq_sample = self.transform(seqs_token =seq_sample, 
                                    seq_dict=self.seq_dict, 
                                    max_seq_len=self.max_seq_len,
                                    code0=code0,
                                    code1=code1,
                                    code2=code2,
                                    )
        return mask_seq_sample

    def __len__(self):
        return self.n_samples
    


# test_dataset = seq_dataset(data_path="applied_dl_bert_impl/data/imgt_I_domain_1_2.csv", seqs_range = [0,1,2])
# print(test_dataset[:3])