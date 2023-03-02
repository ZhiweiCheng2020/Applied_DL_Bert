import torch
import numpy as np


# The function computes loss for train/val/test set
def compute_loss(mask_pos_data, mask_token_data, MaskedLM, 
                 code0, code0_pred,
                 code1, code1_pred,
                 loss_fun):
    # (batch_size, n_mask) --> (batch_size,n_mask,vocab_size)
    masked_pos = mask_pos_data.unsqueeze(-1).expand(-1, -1, MaskedLM.size(-1))
    # (batch_size,source_len,vocab_size) --> (batch_size,n_mask,vocab_size)
    MaskedLM = torch.gather(MaskedLM, 1, masked_pos)
    # calculate the training loss       
    loss_maskLM = loss_fun(MaskedLM.transpose(1,2), mask_token_data)
    loss_code0 = loss_fun(code0_pred, code0.float())
    loss_code1 = loss_fun(code1_pred, code1.float())
    # loss_code2 = loss_fun(code2_pred, code2.float())
    # curr_loss = loss_maskLM + loss_code0
    curr_loss = loss_maskLM + loss_code0 + loss_code1
    return curr_loss, loss_maskLM, loss_code0, loss_code1


# This function defines an EarlyStopper for training, it stops training,
# when the validation loss keeps increasing by at least {min_delta} for {patience} times
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False