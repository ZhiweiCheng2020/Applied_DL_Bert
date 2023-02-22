import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

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
    def __init__(self, ebd_dim, num_head, dropout_rate=0.1):
        super().__init__()
        self.ebd_dim = ebd_dim
        self.num_head = num_head
        self.head_dim = ebd_dim // num_head
        self.dropout_rate = dropout_rate
        assert self.head_dim * num_head == self.ebd_dim, "ebd_dim/num_head should be an integer!"
        
        # Q_linear, K_linear, V_linear: (ebd_dim, ebd_dim)
        self.Q_linear = nn.Linear(ebd_dim, ebd_dim)
        self.K_linear = nn.Linear(ebd_dim, ebd_dim)
        self.V_linear = nn.Linear(ebd_dim, ebd_dim)
        self.attention_linear = nn.Linear(ebd_dim, ebd_dim)
        
    def forward(self, Q0, K0, V0, padding_mask, training):
        # (batch_size, seq_len, ebd_dim)*(ebd_dim,ebd_dim)=(batch_size, seq_len,head_dim*num_head)
        Q = self.Q_linear(Q0)
        K = self.K_linear(K0)
        V = self.V_linear(V0)
        
        batch_size, seq_len, ebd_dim = Q0.size()
        _, source_len, _ = K0.size()
        # self.head_dim
        
        #(self.num_head*batch_size,seq_len or source_len,self.head_dim)  
        Q = Q.view(self.num_head*batch_size, seq_len, self.head_dim)
        K = K.view(self.num_head*batch_size, source_len, self.head_dim)
        V = V.view(self.num_head*batch_size, source_len, self.head_dim)
        
        # batch matrix-matrix product of matrices, normalized
        attention_weight = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.head_dim)
        # (self.num_head*batch_size,seq_len,self.head_dim) * (self.num_head*batch_size,self.head_dim,source_len)
        # --> (self.num_head*batch_size,seq_len,source_len)

        # add padding_mask: (batch_size, source_len)
        assert batch_size, source_len == padding_mask.size()
        # print("padding is okay.")
        # print("attention_weight",attention_weight.shape)
        attention_weight = attention_weight.view(batch_size, self.num_head, seq_len, source_len)
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2) #(batch_size, 1, 1, source_len)
        # pay -inf attention on the [PAD] tokens
        attention_weight = attention_weight.masked_fill(mask=padding_mask, value=float("-inf"))
        attention_weight = attention_weight.view(-1, seq_len, source_len) 
        #(self.num_head*batch_size,seq_len,source_len)
        
        # apply softmax and dropout
        attention_weight = F.softmax(attention_weight, dim=-1) #apply softmax along last dim
        attention_weight = F.dropout(attention_weight, self.dropout_rate, training=training) # apply dropout only when training
        attention = torch.bmm(attention_weight, V)
        # (self.num_head*batch_size,seq_len,source_len) * (self.num_head*batch_size,source_len,self.head_dim)
        # --> (self.num_head*batch_size,seq_len,self.head_dim)
        attention = attention.transpose(0, 1).contiguous().view(seq_len, batch_size, self.num_head*self.head_dim)
        attention = attention.view(batch_size, seq_len, self.num_head*self.head_dim) # the same dim as ebd
        
        Z = self.attention_linear(attention) # linear combination of multiple z, (batch_size, seq_len, self.num_head*self.head_dim)
        # print("Z.shape: ", Z.shape)
        return Z

class Encoder_Layer(nn.Module):
    def __init__(self, ebd_dim, num_head, dim_feedforward, dropout_rate=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(ebd_dim=ebd_dim, num_head=num_head,
                                         dropout_rate=dropout_rate)
        
        # define other sublayers in encoder
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.linear1 = nn.Linear(ebd_dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, ebd_dim)

        self.norm1 = nn.LayerNorm(ebd_dim)
        self.norm2 = nn.LayerNorm(ebd_dim)
        self.gelu = nn.GELU()
    
    # @staticmethod
    # # gelu activation function
    # def gelu(x):
    #     return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))
    
    def forward(self, source_input, training, source_pad_mask=None):
        """
        source_input: [batch_size,source_len,ebd_dim]
        source_pad_mask: [batch_size, source_len]
        """
        
        # Multi-head attention 
        source_attn = self.attention(source_input, source_input, source_input, 
                              padding_mask=source_pad_mask, training=training) #[batch_size,source_len,ebd_dim]
 
        # Residual Dropout 1
        source_input = source_input + self.dropout1(source_attn)  # residual connection
        source_input = self.norm1(source_input)   #[batch_size,source_len,ebd_dim]

        # Feed Forward
        tmp = source_input
        source_input = self.gelu(self.linear1(source_input))  # [source_len,batch_size,dim_feedforward]
        source_input = self.linear2(source_input)  # #[batch_size,source_len,ebd_dim]
        
        # Residual Dropout 2
        source_input = tmp + self.dropout2(source_input)
        source_input = self.norm2(source_input)
        return source_input  # #[batch_size,source_len,ebd_dim]

class Encoder(nn.Module):
    def __init__(self, Encoder_Layer, num_layer):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(Encoder_Layer) for _ in range(num_layer)])
        self.num_layer = num_layer

    def forward(self, source_input, training, source_pad_mask=None):
    # connection of multiple encoder layer
        for sub_layer in self.layers:
            source_input = sub_layer.forward(source_input=source_input, training=training, 
                         source_pad_mask=source_pad_mask)

        return source_input  # #[batch_size,source_len,ebd_dim]
    
class Bert(nn.Module):
    def __init__(self, ebd_dim, vocab_size, max_seq_len, num_head, dim_feedforward,
                 num_layer, ):
        super().__init__() 
        self.ebd_model = input_embedding(ebd_dim=ebd_dim, 
            vocab_size=vocab_size, max_seq_len=max_seq_len)
        self.Encoder_Layer_model = Encoder_Layer(ebd_dim=ebd_dim, num_head=num_head,
                        dim_feedforward=dim_feedforward,)
        self.Encoder_model = Encoder(Encoder_Layer=self.Encoder_Layer_model, 
                             num_layer=num_layer)
        self.linear = nn.Linear(ebd_dim, vocab_size, bias=False)
        self.linear0 = nn.Linear(vocab_size, 3, bias=False)
        self.linear1 = nn.Linear(vocab_size, 1, bias=False)
        self.linear2 = nn.Linear(vocab_size, 1, bias=False)
        

    def forward(self, input, padding_mask, seg=None, training=True):
        ebd = self.ebd_model(input=input, seg=seg) #(batch_size,source_len,ebd_dim)
        encoder = self.Encoder_model(source_input=ebd,training=training,
                        source_pad_mask=padding_mask)  #(batch_size,source_len,ebd_dim)
        # (batch_size,source_len,ebd_dim) --> (batch_size,source_len,vocab_size)
        encoder = self.linear(encoder)
        encoder = nn.LogSoftmax(dim=-1)(encoder) #(batch_size,source_len,vocab_size)
        
        NSP = encoder[:,0,:] # for protein classification
        MaskedLM = encoder[:,1:,:] # for masked token prediction
        
        code0_pred = nn.LogSoftmax(dim=-1)(self.linear0(NSP)) # code0: 3-class
        code1_pred = self.linear1(NSP) # code1: regression
        code2_pred = self.linear2(NSP) # code2: regression
        
        return MaskedLM, code0_pred, code1_pred, code2_pred
