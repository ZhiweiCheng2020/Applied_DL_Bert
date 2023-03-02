import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(42)
import copy
torch.random.manual_seed(42)
g = torch.Generator()
g.manual_seed(42)

class input_embedding(nn.Module):
    def __init__(self, ebd_dim, vocab_size, max_seq_len, num_seg=2) -> None:
        super().__init__()
        self.token_ebd = nn.Embedding(vocab_size, ebd_dim) #number of unique tokens
        self.seg_ebd = nn.Embedding(num_seg, ebd_dim) # not needed in our case
        self.pos_ebd = nn.Embedding(max_seq_len, ebd_dim) # max length of seq
        self.norm = nn.LayerNorm(ebd_dim)
        
    def forward(self, input, seg = None):
        input_len = input.shape[-1] # input sequence length
        position = torch.arange(input_len, dtype=torch.long) # initialize position embedding: [0:181]
       
        position = position.unsqueeze(0).expand_as(input)  # (,input_len) -> (batch_size, input_len)
        if seg:
            Bert_ebd = self.token_ebd(input) + self.pos_ebd(position) + self.seg_ebd(seg)
        else:
            Bert_ebd = self.token_ebd(input) + self.pos_ebd(position)
            
        Bert_ebd = self.norm(Bert_ebd)
        return Bert_ebd
    
class MultiHeadAttention(nn.Module):
    def __init__(self, ebd_dim, num_head):
        super().__init__()
        self.ebd_dim = ebd_dim
        self.num_head = num_head
        self.head_dim = ebd_dim // num_head
        assert self.head_dim * num_head == self.ebd_dim, "single-head embedding length should be an integer!"
        
        # Q_linear, K_linear, V_linear: (ebd_dim, ebd_dim)
        self.Q_linear = nn.Linear(ebd_dim, ebd_dim)
        self.K_linear = nn.Linear(ebd_dim, ebd_dim)
        self.V_linear = nn.Linear(ebd_dim, ebd_dim)
        self.atten_linear = nn.Linear(ebd_dim, ebd_dim)
        
    def forward(self, W_Q, W_K, W_V, padding_mask):
        batch_size, seq_len, _ = W_Q.size()
        # (batch_size, seq_len, num_head, head_dim)
        Q = self.Q_linear(W_Q).view(batch_size, seq_len, self.num_head, self.head_dim)
        K = self.K_linear(W_K).view(batch_size, seq_len, self.num_head, self.head_dim)
        V = self.V_linear(W_V).view(batch_size, seq_len, self.num_head, self.head_dim)
        
        #(num_head, batch_size,seq_len,head_dim)
        Q = Q.view(self.num_head, batch_size, seq_len, self.head_dim)
        K = K.view(self.num_head, batch_size, seq_len, self.head_dim)
        V = V.view(self.num_head, batch_size, seq_len, self.head_dim)
        
        # (num_head, batch_size,seq_len,seq_len)
        atten_w = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.head_dim) 

        # add padding_mask: (batch_size, seq_len)
        assert batch_size, seq_len == padding_mask.size()
        padding_mask = padding_mask.unsqueeze(0).unsqueeze(2) #(1, batch_size, 1, seq_len)
        # padding_mask = padding_mask.unsqueeze(0).unsqueeze(3) #(1, batch_size, seq_len, 1), masking both rows and columns? 
        # pay -inf attention on the [PAD] tokens
        atten_w = atten_w.masked_fill(mask=padding_mask, value=float("-inf"))

        # apply softmax and dropout
        atten_w = F.softmax(atten_w, dim=-1) #apply softmax along last dim
        atten_w = torch.matmul(atten_w, V) #(num_head, batch_size,seq_len,head_dim)
        atten_w = atten_w.view(batch_size, seq_len, self.num_head, self.head_dim)
        atten_w = atten_w.view(batch_size, seq_len, self.ebd_dim)
        
        Z = self.atten_linear(atten_w) # linear combination of multiple z, (batch_size, seq_len, ebd_dim)
        return Z

class Encoder_Layer(nn.Module):
    def __init__(self, ebd_dim, num_head, feedforward_dim, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(ebd_dim=ebd_dim, num_head=num_head)
        
        # define other sublayers in encoder
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.linear1 = nn.Linear(ebd_dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, ebd_dim)

        self.norm1 = nn.LayerNorm(ebd_dim)
        self.norm2 = nn.LayerNorm(ebd_dim)
        self.gelu = nn.GELU()
    
    # @staticmethod
    # # gelu activation function
    # def gelu(x):
    #     return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    
    def forward(self, input_ebd, pad_mask=None):
        # Multi-head attention 
        # (batch_size,seq_len,ebd_dim)
        attn = self.attention(input_ebd, input_ebd, input_ebd, 
                              padding_mask=pad_mask) 
 
        # Residual Dropout 1
        input_ebd = input_ebd + self.dropout1(attn)  # residual connection
        input_ebd = self.norm1(input_ebd)

        # Feed Forward
        tmp = input_ebd
        input_ebd = self.gelu(self.linear1(input_ebd))  # (batch_size,seq_len,ebd_dim) -> (batch_size,seq_len,feedforward_dim)
        input_ebd = self.linear2(input_ebd)  # (batch_size,seq_len,feedforward_dim) -> (batch_size,seq_len,ebd_dim)
        
        # Residual Dropout 2
        input_ebd = tmp + self.dropout2(input_ebd)
        input_ebd = self.norm2(input_ebd)
        return input_ebd  # (batch_size,seq_len,ebd_dim)

class Encoder(nn.Module):
    def __init__(self, Encoder_Layer, num_layer):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(Encoder_Layer) for _ in range(num_layer)]) # deepcopy all encoder layers
        self.num_layer = num_layer

    def forward(self, input_ebd, pad_mask=None):
    # connection of multiple encoder layer
        for sub_layer in self.layers:
            input_ebd = sub_layer.forward(input_ebd=input_ebd,
                         pad_mask=pad_mask)

        return input_ebd  # (batch_size,seq_len,ebd_dim)
    
class Bert(nn.Module):
    def __init__(self, ebd_dim, vocab_size, max_seq_len, num_head, feedforward_dim,
                 num_layer, ):
        super().__init__() 
        self.ebd_model = input_embedding(ebd_dim=ebd_dim, 
            vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.Encoder_Layer_model = Encoder_Layer(ebd_dim=ebd_dim, num_head=num_head,
                        feedforward_dim=feedforward_dim,)
        self.Encoder_model = Encoder(Encoder_Layer=self.Encoder_Layer_model, 
                             num_layer=num_layer)
        self.linear = nn.Linear(ebd_dim, vocab_size, bias=False)
        self.linear0 = nn.Linear(vocab_size, 3, bias=False) # for code0: A/B/C
        self.linear1 = nn.Linear(vocab_size, 64, bias=False) # for code1: 64 types
        self.linear2 = nn.Linear(vocab_size, 1027, bias=False) # for code2: 1027 types
        

    def forward(self, input, padding_mask, seg=None):
        ebd = self.ebd_model(input=input, seg=seg) #(batch_size,seq_len,ebd_dim)
        encoder = self.Encoder_model(input_ebd=ebd,
                        pad_mask=padding_mask)  #(batch_size,seq_len,ebd_dim)
        # (batch_size,seq_len,ebd_dim) --> (batch_size,seq_len,vocab_size)
        encoder = self.linear(encoder)
        encoder = nn.LogSoftmax(dim=-1)(encoder) #(batch_size,seq_len,vocab_size)
        
        NSP = encoder[:,0,:] # for protein classification
        MaskedLM = encoder[:,1:,:] # for masked token prediction
        
        code0_pred = nn.LogSoftmax(dim=-1)(self.linear0(NSP)) # code0: 3-class
        code1_pred = nn.LogSoftmax(dim=-1)(self.linear1(NSP)) # code1: 63-class
        code2_pred = nn.LogSoftmax(dim=-1)(self.linear2(NSP)) # code2
        
        return MaskedLM, code0_pred, code1_pred, code2_pred


