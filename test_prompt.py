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
from data_loader_prompt import Dataload_prompt
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
    def __init__(self, model_name):
        super(prompt, self).__init__()
        self.model_name = model_name
    
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        self.special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
        self.num_added_toks = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        #self.tokenizer.all_special_tokens
        #self.tokenizer.all_special_ids
        
    def forward(self, turns, prompts):
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        for i,turn in enumerate(turns):
            
            
            if self.model_name == 'RoBERTa_Prompt':
                prompt_turn = turn + '. it is <mask>.'
            elif self.model_name == 'RoBERTa_Prompt_dem':
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
            masked_tokens.append( self.tokenizer.encode(self.tokenizer.tokenize(prompts[i])[0])[1])
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
    
class prompt_MLM(nn.Module):
    def __init__(self):
        super(prompt_MLM, self).__init__()
        self.max_pred = 20
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        self.special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
        self.num_added_toks = self.tokenizer.add_special_tokens(self.special_tokens_dict)
        #self.tokenizer.all_special_tokens
        #self.tokenizer.all_special_ids 
        
    def forward(self, turns, prompts):  
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        
        for i,turn in enumerate(turns):
            prompt_turn = turn + '. it is '  + prompts[i] + '.'
            
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
            
            #Mask 15% of tokens within utterance
            
            #Find where utterance is terminated
            dot_value = self.tokenizer.encode(self.tokenizer.tokenize('.'),add_special_tokens=False)[0]
            #Input without prompt, trancate the utterance
            input_id_tranc = input_ids[:((input_ids == dot_value).nonzero(as_tuple=True)[0])[0].item()]
            #Cantidated mask tokens
            n_pred =  min(self.max_pred, max(1, int(round(len(tokens) * 0.15)))) # 15 % of tokens in one sequence
            cand_maked_pos = [i for i, token in enumerate(input_id_tranc) if token!=0 and token!=1 and token!=2]
            shuffle(cand_maked_pos)
            
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)
                masked_tokens.append(input_ids[pos].item())
                input_ids[pos] = self.tokenizer.mask_token_id
                
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)  
            
            
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
    
    #if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem' :
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
    
    #Prepare data
    df_test = pd.read_csv('data/text/test.csv')
    df_train = pd.read_csv('data/text/train.csv')  
    if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_inverse':
        test_data = Dataload(df_test)
    elif  args.model_name == 'RoBERTa_Prompt_dem':   
        test_data = Dataload_prompt(df_test,df_train,'test')
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=True,drop_last=False)
    
    tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
    special_tokens_dict = {'additional_special_tokens': ['anomia','disfluency','agrammatism','fluent']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    #if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem':   
    if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem':   
         features = prompt(args.model_name)
    elif args.model_name == 'RoBERTa_Prompt_inverse':
         features = prompt_MLM()
    
    model.eval()
    test_loss = 0
    test_pred = []
    test_labels = []   
    
    with torch.no_grad():
        for step, (turns, labels, prompts) in tqdm(enumerate(test_loader)):
            
            _input,_mask, _mlm_tokens,_mlm_pos  = features(turns,prompts)
            _labels = labels
            
            if torch.cuda.device_count() > 1:
                _input = _input.cuda(non_blocking=True)
                _mask  = _mask.cuda(non_blocking=True)
                _mlm_tokens  = _mlm_tokens.cuda(non_blocking=True)
                _mlm_pos  = _mlm_pos.cuda(non_blocking=True)
                _labels = _labels.cuda(non_blocking=True)
            else:    
                _input = _input.to(device)
                _mask  = _mask.to(device)
                _mlm_tokens  = _mlm_tokens.to(device)
                _mlm_pos  = _mlm_pos.to(device)
                _labels = _labels.to(device)
            
            output = model(_input, _mask, _mlm_tokens, _mlm_pos) 
            loss = loss_fn(output.view(-1, 50269),_mlm_tokens.view(-1))
            
            
            
            test_loss+=loss.item()
            
            # calculate preds
            if args.model_name == 'RoBERTa_Prompt' or args.model_name == 'RoBERTa_Prompt_dem': 
                predictions = []
                maxs = torch.argmax(output.cpu().detach(), dim=-1)
                for max in maxs:
                    predictions.append(tokenizer.decode(max))
                
                for pred in predictions:

                    if pred == label_names[0]:
                        test_pred.append(0)
                    elif pred == label_names[1]:   
                        test_pred.append(1)
                    elif pred == label_names[2]:
                        test_pred.append(2)
                    elif pred == label_names[3]:
                        test_pred.append(3)
                    else:
                        test_pred.append(4)
                
                
            elif args.model_name == 'RoBERTa_Prompt_inverse':
                for i in range (0,_input.size(0)):
                    prompt_label = tokenizer.encode(tokenizer.tokenize(prompts[i]),add_special_tokens=False)[0] 
                    prompt_pos = ((_input[i] == prompt_label).nonzero(as_tuple=True)[0])[0].item()
                    min_loss = []
                    for k in range (0,len(label_names)-1): 
                        prompt_new = tokenizer.encode(tokenizer.tokenize(label_names[k]),add_special_tokens=False)
                        _input[i][prompt_pos] = torch.tensor(prompt_new)
                        
                        output = model(_input[i].unsqueeze(dim=0), _mask[i].unsqueeze(dim=0), _mlm_tokens[i].unsqueeze(dim=0), _mlm_pos[i].unsqueeze(dim=0)) 
                        loss = loss_fn(output.view(-1, 50269),_mlm_tokens[i].view(-1))
                        min_loss.append(loss.item())
                    test_pred.append(min_loss.index(min(min_loss)))    
                
            test_labels.extend(_labels.cpu().detach().numpy())  
    
    print("")
    print('Test resuts') 
    test_loss = test_loss/(step+1)
    print("Test loss: %0.3f"%(test_loss))
    print('Classifiation report')
    print(classification_report(test_pred, test_labels, target_names=label_names,  labels=[0,1,2,3,4], zero_division=0,digits=3))
    print('Accuracy:')
    matrix = confusion_matrix(test_labels, test_pred,labels=[0,1,2,3,4])
    print(matrix.diagonal()/matrix.sum(axis=1))
    print('Weighted accuracy')
    print(balanced_accuracy_score(test_labels, test_pred))
    print(accuracy_score(test_labels, test_pred))
            

if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    
    USE_CUDA = torch.cuda.is_available()
    test(args)