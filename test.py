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
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import AdamW,Adam
from sklearn.metrics import classification_report,confusion_matrix,balanced_accuracy_score,accuracy_score
from random import shuffle
from models.mlm_fine_tune import RoBERTa_MLM
from models.entailment import RoBERTa_Entailment

# Set the seed value all over the place to make this reproducible.
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)



class features_finetune(nn.Module):
    def __init__(self):
        super(features_finetune, self).__init__()
        
        
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
    def forward(self, turns):
        
        ids = []
        masks = []
        for i,turn in enumerate(turns):
        
            
            tokens = self.tokenizer.tokenize(turn)
           
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
            
            ids.append(input_ids)
            masks.append(attention_mask)
        
            
        ids = torch.stack(ids, dim=0)
        masks = torch.stack(masks, dim=0)
        return ids,masks    
    
    
    
class features_MLM(nn.Module):
    def __init__(self):
        super(features_MLM, self).__init__()
        self.max_pred = 20
        
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
    def forward(self, turns):
        
        ids = []
        masks = []
        mlm_tokens = []
        mlm_pos = []
        for i,turn in enumerate(turns):
        
            
            tokens = self.tokenizer.tokenize(turn)
           
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
            
            #Mask
            n_pred =  min(self.max_pred, max(1, int(round(len(tokens) * 0.15)))) # 15 % of tokens in one sequence
            cand_maked_pos = [i for i, token in enumerate(input_ids) if token!=0 and token!=1 and token!=2]
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
    
class features_Entail(nn.Module):
    def __init__(self):
        super(features_Entail, self).__init__()
        self.tokenizer  = RobertaTokenizer.from_pretrained('roberta-base')
        self.dict = {
              "fluent": "Fluent speech",  
              "anomia": "Talking around words or empty speech",
              "disfluency": "Word repetition or revision",
              "agrammatism": "Agrammatism or paragrammatism in speech"
            }
        
    def forward(self, turns):
        
        
        in_ids = []
        in_mask = []
        for i,turn in enumerate(turns):
            ids = []
            masks = []
            for i in range (len(self.dict)):
                turn_des = turn + '. ' + list(self.dict.values())[i] + '.'
                
                
                tokens = self.tokenizer.tokenize(turn_des)
                
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
                
                ids.append(input_ids)
                masks.append(attention_mask)
            ids = torch.stack(ids, dim=0)
            masks = torch.stack(masks, dim=0)
            in_ids.append(ids)
            in_mask.append(masks)
            
        in_ids = torch.stack(in_ids, dim=0)
        in_mask = torch.stack(in_mask, dim=0)
        
        return in_ids,in_mask 
    
def test(args):
    label_names = ["fluent","anomia","disflueny","agrammatism"]
    model_bin = [args.task, args.model_name, args.batch_size, args.lr, 'train' ]  
    
    #Load model
    if args.model_name == 'RoBERTa':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base",num_labels=args.num_classes,output_attentions=False,output_hidden_states=False)
    elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM'  or args.model_name == 'RoBERTa_CLS':
        print('ok')
        model = RoBERTa_MLM()
    elif args.model_name == 'RoBERTa_entail':   
        model = RoBERTa_Entailment()    
        
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
        if args.model_name == 'RoBERTa' or args.model_name == 'RoBERTa_entail':
            loss_fn = nn.CrossEntropyLoss().cuda(int(args.cuda))
        elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS':
            loss_ml = nn.CrossEntropyLoss(ignore_index = 0).cuda(int(args.cuda))
            loss_clf = nn.CrossEntropyLoss().cuda(int(args.cuda))
    else:
        print('You use only one device')  
        device = torch.device("cuda" if USE_CUDA else "cpu")
        if os.path.exists(os.path.join(args.checkpoint_path,  '_'.join([str(i) for i in model_bin]))):
            model = torch.load(os.path.join(args.checkpoint_path, '_'.join([str(i) for i in model_bin])))
            print('Pretrained model %s has been loaded'%(os.path.join(args.checkpoint_path,  '_'.join([str(i) for i in model_bin]))))
        else:
            print('Pretrained model does not exist!!!')
        model.to(device)
        if args.model_name == 'RoBERTa' or args.model_name == 'RoBERTa_entail':
            loss_fn = nn.CrossEntropyLoss()
        elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS':   
            loss_ml = nn.CrossEntropyLoss(ignore_index = 0)
            loss_clf = nn.CrossEntropyLoss()
        
    #Prepare data
    df_test = pd.read_csv('data/text/test.csv')
    test_data = Dataload(df_test)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size,shuffle=True,drop_last=False)
    
    if args.model_name == 'RoBERTa':
        features = features_finetune()
    elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS': 
        features = features_MLM()
    elif args.model_name == 'RoBERTa_entail':
        features =  features_Entail()      
    
    model.eval()
    test_loss = 0
    test_pred = []
    test_labels = []   
    
    with torch.no_grad():
        for step, (turns, labels, prompts) in tqdm(enumerate(test_loader)):
            if args.model_name == 'RoBERTa' or args.model_name == 'RoBERTa_entail':
                _input, _mask = features(turns)
                _labels = labels
                
                if torch.cuda.device_count() > 1:
                    _input = _input.cuda(non_blocking=True)
                    _mask  = _mask.cuda(non_blocking=True)
                    _labels = _labels.cuda(non_blocking=True)
                else:    
                    _input = _input.to(device)
                    _mask  = _mask.to(device)
                    _labels = _labels.to(device)
                    
                output = model(input_ids = _input, attention_mask=_mask) 
                if args.model_name == 'RoBERTa':
                    loss = loss_fn(output[0],_labels)
                elif args.model_name == 'RoBERTa_entail':
                    loss = loss_fn(output,_labels)
                
                    
            elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS':     
                _input,_mask, _mlm_tokens,_mlm_pos  = features(turns)
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
                    
                output = model(_input, _mask, _mlm_pos, _mlm_pos)  
                
                loss_1 = loss_clf(output[0],_labels)
                loss_2 = loss_ml(output[1].view(-1, 50265),_mlm_tokens.view(-1))
                
                if  args.model_name == 'RoBERTa_Multitask':
                    loss = 0.5*loss_1 + 0.5*loss_2
                elif args.model_name == 'RoBERTa_MLM':    
                    loss = 0*loss_1 + 1*loss_2
                elif args.model_name == 'RoBERTa_CLS':   
                    loss = 1*loss_1 + 0*loss_2
            
            
            
            test_loss+=loss.item()
            
            # calculate preds
            if args.model_name == 'RoBERTa_entail':
                test_pred.extend(np.argmax(output.cpu().detach().numpy(),axis=-1))
            else:
                test_pred.extend(np.argmax(output[0].cpu().detach().numpy(),axis=-1))
                
            test_labels.extend(_labels.cpu().detach().numpy())

    print("")
    print('Test resuts') 
    test_loss = test_loss/(step+1)
    print("Test loss: %0.3f"%(test_loss))
    print('Classifiation report')
    print(classification_report(test_pred, test_labels, target_names=label_names,digits=3))
    print('Accuracy:')
    matrix = confusion_matrix(test_labels, test_pred)
    print(matrix.diagonal()/matrix.sum(axis=1))
    print('Weighted accuracy')
    print(balanced_accuracy_score(test_labels, test_pred))
    print(accuracy_score(test_labels, test_pred))

    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
   
    USE_CUDA = torch.cuda.is_available()
    test(args)    

