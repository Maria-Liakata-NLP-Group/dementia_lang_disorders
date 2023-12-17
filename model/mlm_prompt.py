from transformers import  RobertaTokenizer,RobertaModel
import torch
import os
import torch.nn as nn
import math

class RoBERTa_MLM_Prompt(torch.nn.Module):
    def __init__(self,d_model=768, voc_size = 50269):
        super(RoBERTa_MLM_Prompt, self).__init__()
        self.d_model = d_model
        self.voc_size = voc_size
        self.model =  RobertaModel.from_pretrained("roberta-base")
        
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        self.special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
        self.num_added_toks = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        
        self.model.resize_token_embeddings(len(self.tokenizer))
        

        self.dense_mlm = nn.Linear(self.d_model, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model,eps=1e-05)
        self.decoder = nn.Linear(self.d_model, self.voc_size, bias=False)
     
        
    def forward(self,  _input,_mask, _mlm_tokens,_mlm_pos ):
        #Feeding the input to BERT model to obtain contextualized representations
        output = self.model(input_ids = _input, attention_mask=_mask)
        hidden_state = output[0]
        
        
        #**** Mask Langugage Model ****
        
        masked_pos = _mlm_pos[:, :, None].expand(-1, -1, self.d_model) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(hidden_state, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        
        h_masked = self.dense_mlm(h_masked) # [batch_size, max_pred, d_model]
        h_masked = self.layer_norm(h_masked)
        logits_lm = self.decoder(h_masked) # [batch_size, max_pred, vocab_size]
        
        return logits_lm
    
     