from transformers import  RobertaTokenizer,RobertaModel
import torch
import os
import torch.nn as nn
import math

class RoBERTa_MLM(torch.nn.Module):
    def __init__(self,d_model=768, classes=4,voc_size = 50265):
        super(RoBERTa_MLM, self).__init__()
        self.d_model = d_model
        self.classes = classes
        self.voc_size = voc_size
        self.model =  RobertaModel.from_pretrained("roberta-base")
        
        self.dense = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.1, inplace=False)
        )
        
       
        
        self.proj = nn.Linear(self.d_model, self.classes)
        
        self.dense_mlm = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model,eps=1e-05)
        self.decoder = nn.Linear(self.d_model, self.voc_size, bias=False)
     
        
    def forward(self,  _input,_mask, mlm_tokens,mlm_pos ):
        #Feeding the input to BERT model to obtain contextualized representations
        output = self.model(input_ids = _input, attention_mask=_mask)
        
        #**** Classificaiton ****
        
        #Obtaining the representation of [CLS] head
        hidden_state = output[0]
        cls_rep = hidden_state[:, 0]
        
        h_pooled = self.dense(cls_rep) # [batch_size, d_model]
        logits_cls = self.proj(h_pooled) # [batch_size, 4] 
        
        
        
        #**** Mask Langugage Model ****
        
        masked_pos = mlm_pos[:, :, None].expand(-1, -1, self.d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(hidden_state, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        
        h_masked = self.dense_mlm(h_masked) # [batch_size, max_pred, d_model]
        h_masked = self.layer_norm(h_masked)
        logits_lm = self.decoder(h_masked) # [batch_size, max_pred, vocab_size]
        
        return logits_cls, logits_lm
    
     