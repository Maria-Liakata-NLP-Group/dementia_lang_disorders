import os
from tqdm import tnrange, tqdm
import pandas as pd
from argument_parser import parse_arguments
import torch
import random
import numpy as np
from transformers import get_linear_schedule_with_warmup
from transformers import  RobertaTokenizer,RobertaForSequenceClassification
from data_loader import Dataload
from data_loader_prompt_markers import Dataload_prompt
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW,Adam
from sklearn.metrics import classification_report,confusion_matrix,balanced_accuracy_score,accuracy_score
from random import shuffle
from models.mlm_prompt import RoBERTa_MLM_Prompt
import torch.nn.functional as F

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


class prompt(nn.Module):
    def __init__(self):
        super(prompt, self).__init__()
    
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        self.special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
        self.num_added_toks = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        #self.tokenizer.all_special_tokens
        #self.tokenizer.all_special_ids
        
    def forward(self, turns):
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        for i,turn in enumerate(turns):
           
            prompt_turn = turn #Template formed in dataLoad
        
            
            tokens = self.tokenizer.tokenize(prompt_turn)
           
            encoded_dict = self.tokenizer.encode_plus(
                tokens,  # document to encode.
                add_special_tokens=True,  # add '[CLS]' and '[SEP]'
                padding='max_length',  # set max length
                truncation=True,  # truncate longer messages
                pad_to_max_length=True,  # add padding
                return_attention_mask=True,  # create attn. masks
                return_tensors='pt'  # return pytorch tensors
            )
            
            input_ids = encoded_dict['input_ids'].squeeze(dim=0)
            attention_mask = encoded_dict['attention_mask'].squeeze(dim=0)
            
            masked_tokens, masked_pos = [], []
            #if self.model_name == 'RoBERTa_Prompt' or self.model_name == 'RoBERTa_Prompt_dem':
            masked_tokens.append( self.tokenizer.encode(self.tokenizer.tokenize('<mask>')[0])[1])
            masked_pos.append((input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0])
            #elif self.model_name == 'RoBERTa_decouple':
            #    for _prompt in prompts[i].split(','):
            #        masked_tokens.append(self.tokenizer.encode(self.tokenizer.tokenize(_prompt),add_special_tokens=False))
            #        masked_pos.append((input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0])
                
                
            ids.append(input_ids)
            masks.append(attention_mask)
            mlm_tokens.append(torch.tensor(masked_tokens))
            mlm_pos.append(torch.tensor(masked_pos))
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        mlm_tokens = torch.stack(mlm_tokens, dim=0)
        mlm_pos = torch.stack(mlm_pos, dim=0)
        return ids,masks,mlm_tokens,mlm_pos 

def test(args):
    label_names = ["fluent","anomia","disfluency","agrammatism", "none"]
    model_bin = [args.task, args.model_name, args.batch_size, args.lr, 'train' ]
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc', 'lr'])
    
    model = RoBERTa_MLM_Prompt()
    
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0])
        
        if os.path.exists(os.path.join(args.checkpoint_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.checkpoint_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model has been loaded')
        else:
            print('Pretrained model does not exist!!!')
        
        #Run the model on a specified GPU and run the operations to multiple GPUs
        torch.cuda.set_device(int(args.cuda))
        model.cuda(int(args.cuda))
        if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem': 
            loss_fn = nn.CrossEntropyLoss().cuda(int(args.cuda))
        elif args.model_name == 'RoBERTa_Prompt_inverse':
            loss_fn = nn.CrossEntropyLoss(ignore_index = 0).cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.checkpoint_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.checkpoint_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.checkpoint_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
            
        model.to(device)
        if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem': 
            loss_fn = nn.CrossEntropyLoss()
        elif args.model_name == 'RoBERTa_Prompt_inverse':
            loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
    
    df_train = pd.read_csv('data/text/train.csv')  
    
    
    tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
    special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    #if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem':   
    features = prompt() 
         
  
    
    model.eval()
    test_loss = 0
    test_pred = []
    test_labels = []  
    
    
    
    with torch.no_grad():
        df = pd.DataFrame(columns = ['Client', 'Session', 'Cohort', 'Anomia', 'Disfluency', 'Agrammatism', 'Fluency'])
        index = 0
        
        for subdir , dirs, files in os.walk('data/text/longtitudinal'): 
             for file in tqdm(files):
                 if file.endswith(".txt"):
                     df_test = pd.read_csv(os.path.join('data/text/longtitudinal', file), header=None)
                     df_test.columns = ["Turn"]
                     test_data = Dataload_prompt(df_test,df_train)
                     test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=True,drop_last=False)
                     
                     subject = file.split('_')[1].split('.')[0].split('-')[0]
                     session = file.split('_')[1].split('.')[0].split('-')[1]
                     cohort = file.split('_')[0]
                     
                     score_anomia = 0.0
                     score_disfluency = 0.0
                     score_agrammatism = 0.0
                     score_fluent = 0.0
                     for step, turns in tqdm(enumerate(test_loader)):
                     
                         _input,_mask, _mlm_tokens,_mlm_pos  = features(turns)
                         label_ids = []
                         label_ids.append(tokenizer.encode(tokenizer.tokenize('anomia')[0])[1])
                         label_ids.append(tokenizer.encode(tokenizer.tokenize('disfluency')[0])[1])
                         label_ids.append(tokenizer.encode(tokenizer.tokenize('agrammatism')[0])[1])
                         label_ids.append(tokenizer.encode(tokenizer.tokenize('fluent')[0])[1])
                         
                         label_ids = torch.tensor(label_ids)
                         
                         if torch.cuda.device_count() > 1:
                             _input = _input.cuda(non_blocking=True)
                             _mask  = _mask.cuda(non_blocking=True)
                             _mlm_tokens  = _mlm_tokens.cuda(non_blocking=True)
                             _mlm_pos  = _mlm_pos.cuda(non_blocking=True)
                             label_ids  = label_ids.cuda(non_blocking=True)
                         else:    
                             _input = _input.to(device)
                             _mask  = _mask.to(device)
                             _mlm_tokens  = _mlm_tokens.to(device)
                             _mlm_pos  = _mlm_pos.to(device)
                             label_ids  = label_ids.to(device)
                             
                         output = model(_input, _mask, _mlm_tokens, _mlm_pos) 
                         output = output.view(-1, 50269)
                         
                        
                         
                         label_logits = torch.index_select(output, -1, label_ids)
                         label_prob = torch.nn.functional.softmax(label_logits, dim=-1)
                         
                         
                         
                         score_anomia += torch.mean(torch.index_select(label_prob, -1, ((label_ids == label_ids[0].item()).nonzero(as_tuple=True)[0]))).item()
                         score_disfluency += torch.mean(torch.index_select(label_prob, -1, ((label_ids == label_ids[1].item()).nonzero(as_tuple=True)[0]))).item()
                         score_agrammatism += torch.mean(torch.index_select(label_prob, -1, ((label_ids == label_ids[2].item()).nonzero(as_tuple=True)[0]))).item()
                         score_fluent += torch.mean(torch.index_select(label_prob, -1, ((label_ids == label_ids[3].item()).nonzero(as_tuple=True)[0]))).item()
                         
                        
                         
                     score_anomia = score_anomia/len(test_loader)
                     score_disfluency = score_disfluency/len(test_loader)
                     score_agrammatism = score_agrammatism/len(test_loader)
                     score_fluent = score_fluent/len(test_loader)
                     
                     df.loc[index] = pd.Series({'Client':str(subject),'Session':str(session) , 'Cohort':str(cohort), 'Anomia':float(score_anomia) , 'Disfluency':float(score_disfluency), 'Agrammatism':float(score_agrammatism), 'Fluency':float(score_fluent)})
                     index +=1
                     
    df.to_csv(os.path.join('data/markers.csv'),header=True)
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    USE_CUDA = torch.cuda.is_available()
    test(args)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    