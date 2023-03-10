import torch
import os
import sys
import random
import numpy as np
from torch.utils.data import DataLoader
# import src.data.data_preprocess as preprocess
# from src.models.utils import compute_loss 
import data.data_preprocess as preprocess
from models.utils import compute_loss 
import pickle


class umap_creator():
    def __init__(self, model_path, batch_size=64):
        self.CE_loss = torch.nn.CrossEntropyLoss()
        self.curr_path = os.path.dirname(os.getcwd())
        print(self.curr_path)
        # load all data
        dataset = preprocess.seq_dataset(data_path=os.path.join(self.curr_path, 
                                                    "data/protein_data.csv"), 
                                            seqs_range = random.sample(range(0, 13478), 13478), 
                                            seed=42)
        self.data_loader = DataLoader(dataset=dataset, 
                                                batch_size=batch_size, 
                                                shuffle=True,)

        # load saved model
        # Bert_model = torch.load(os.path.join(curr_path, "models", "saved_model.pth"))
        self.Bert_model = torch.load(model_path)
        self.Bert_model.eval()

    # a simple function to transform one-hot-encoding to intergers
    @staticmethod
    def one_hot_to_int(one_hot_tensor):
        np_arr = one_hot_tensor.numpy()
        int_version = list(np.argmax(np_arr, axis=1))
        return int_version

    def prepare_umap(self):
        with torch.no_grad():
            for i, (token_data, mask_pos_data, mask_token_data, code0, code1, code2, padding_mask) in enumerate(self.data_loader):
                # remove the extra dimension
                token_data, mask_pos_data, mask_token_data, padding_mask, code0, code1, code2 = token_data.squeeze(1), \
                mask_pos_data.squeeze(1), mask_token_data.squeeze(1), padding_mask.squeeze(1),\
                    code0.squeeze(1), code1.squeeze(1), code2.squeeze(1)
                MaskedLM, code0_pred, code1_pred, _, NSP = self.Bert_model(input=token_data, padding_mask=padding_mask)
                # curr_loss, loss_maskLM, loss_code0, loss_code1 = compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                #                     code0, code0_pred,
                #                     code1, code1_pred,
                #                     loss_fun=self.CE_loss)
                # print(curr_loss, loss_maskLM, loss_code0, loss_code1)
                
                if i == 0:
                    NSP_ebd = NSP
                    code0_int = self.one_hot_to_int(code0)
                    code1_int = self.one_hot_to_int(code1)
                else:
                    NSP_ebd = torch.cat((NSP_ebd, NSP), 0)
                    code0_int.extend(self.one_hot_to_int(code0))
                    code1_int.extend(self.one_hot_to_int(code1))

        # save embedding for future visulisation
        self.bert_ebd_path = os.path.join(self.curr_path, "models", "bert_ebd.pkl")
        with open(self.bert_ebd_path, "wb") as f:
            pickle.dump([NSP_ebd, code0_int, code1_int], f)    
            print("----UMAP data preparation - complete!-------")
            
        return self.bert_ebd_path

# if __name__ == "__main__":
#     umap_creator()

