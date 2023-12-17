from transformers import  RobertaTokenizer,RobertaModel
import torch
import os
import torch.nn as nn
import math


class RoBERTa_Entailment(torch.nn.Module):
    def __init__(self,d_model=768, classes=2):
        super(RoBERTa_Entailment, self).__init__()
        self.d_model = d_model
        self.classes = classes
        self.model =  RobertaModel.from_pretrained("roberta-base")
        
        self.dense = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout(0.1, inplace=False)
        )
        
       
        
        self.proj = nn.Linear(self.d_model, self.classes)
        
        
    def forward(self,  input_ids,attention_mask):
        
        forwards = input_ids.size(dim=1)
        entailments = []
        for i in range (0,forwards):
            input = input_ids[:,i,:]
            mask = attention_mask[:,i,:]
        
            #Feeding the input to BERT model to obtain contextualized representations
            output = self.model(input_ids = input, attention_mask=mask)
            
            #Obtaining the representation of [CLS] head
            hidden_state = output[0]
            cls_rep = hidden_state[:, 0]
            
            h_pooled = self.dense(cls_rep) # [batch_size, d_model]
            logits_cls = self.proj(h_pooled) # [batch_size, classis] 
            
            entailment_prob = logits_cls[:,0]
            entailments.append(entailment_prob)
            
        logits = torch.stack(entailments, dim=-1)
        
        return logits