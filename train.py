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
from sklearn.metrics import classification_report,confusion_matrix
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
    def __init__(self,  model_name):
        super(features_finetune, self).__init__()
        self.model_name = model_name
        
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
            n_pred =  min(self.max_pred, max(1, int(round(len(tokens) * 0.50)))) # 15 % of tokens in one sequence
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
              "anomia": "Talking around words or empty speech or incomplete speech",
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


def train(args):
    label_names = ["fluent","anomia","disflueny","agrammatism"]
    model_bin = [args.task, args.model_name, args.batch_size, args.lr, 'train' ]
    df = pd.DataFrame(columns = ['Epoch','Tr Loss', 'Val Loss', 'Tr Acc', 'Val Acc', 'lr'])
    
    
    #Load model
    if args.model_name == 'RoBERTa':
        model = RobertaForSequenceClassification.from_pretrained("roberta-base",num_labels=args.num_classes,output_attentions=False,output_hidden_states=False)
    elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM'  or args.model_name == 'RoBERTa_CLS':
        model = RoBERTa_MLM()
    elif args.model_name == 'RoBERTa_entail':   
        model = RoBERTa_Entailment()
        
    if torch.cuda.device_count() > 1:
        print('You use %d GPUs'%torch.cuda.device_count())
        model = nn.DataParallel(model, device_ids=[0,1,2,3])
        if args.model_name == 'RoBERTa_CLS':
            if os.path.exists('tmp/fine-tuning_RoBERTa_MLM_32_1e-05_train'):
                model = torch.load('tmp/fine-tuning_RoBERTa_MLM_32_1e-05_train')
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
        if args.model_name == 'RoBERTa_CLS':
            if os.path.exists('tmp/fine-tuning_RoBERTa_MLM_32_1e-05_train'):
                model = torch.load('tmp/fine-tuning_RoBERTa_MLM_32_1e-05_train')
                print('Pretrained model has been loaded')
            else:
                print('Pretrained model does not exist!!!')
        model.to(device)
        if args.model_name == 'RoBERTa' or args.model_name == 'RoBERTa_entail':
            loss_fn = nn.CrossEntropyLoss()
        elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS':   
            loss_ml = nn.CrossEntropyLoss(ignore_index = 0)
            loss_clf = nn.CrossEntropyLoss()
        
        
    
    #Prepare data
    df_train = pd.read_csv('data/text/train.csv')
    train_data = Dataload(df_train)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size,shuffle=True,drop_last=False)
    
    
    df_val = pd.read_csv('data/text/val.csv')
    val_data = Dataload(df_val)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size,shuffle=True,drop_last=False)
    
    
    if args.model_name == 'RoBERTa':
        features = features_finetune()
    elif args.model_name == 'RoBERTa_Multitask' or args.model_name == 'RoBERTa_MLM' or args.model_name == 'RoBERTa_CLS': 
        features = features_MLM()
    elif args.model_name == 'RoBERTa_entail':
        features =  features_Entail()    
   
    
    #AdamW gets better results than Adam
    optimizer = AdamW([{'params': model.parameters(), 'lr': args.lr}], weight_decay=1.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
             num_warmup_steps=0,
             num_training_steps=len(train_loader)*args.epochs )
    
    
   
    best_loss = 1000
    trigger_times = 0
    
    #train_iterator = tqdm.notebook.tnrange(int(args.epochs), desc="Epoch")
    for epoch in range(0, args.epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        
        print('Training...')
        model.train()
        
        tr_loss = 0.0
        train_labels = []
        train_predict = []
        for step, (turns, labels, prompts) in tqdm(enumerate(train_loader)):
            
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
                    loss = (1/0.5139)*loss_1 + (1/2.4149)*loss_2
                elif args.model_name == 'RoBERTa_MLM':    
                    loss = 0*loss_1 + 1*loss_2
                elif args.model_name == 'RoBERTa_CLS':   
                    loss = 1*loss_1 + 0*loss_2
                
            
            tr_loss += loss.item()
            
            model.zero_grad()
            loss.backward()
            del loss
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
        
            
            # calculate preds
            if args.model_name == 'RoBERTa_entail':
                train_predict.extend(np.argmax(output.cpu().detach().numpy(),axis=-1))
            else:
                train_predict.extend(np.argmax(output[0].cpu().detach().numpy(),axis=-1))
                
            train_labels.extend(_labels.cpu().detach().numpy())
            
            #if (step + 1) % args.statistic_step == 0:
            #    print('classifiation report')
            #    print(classification_report(train_predict, train_labels, target_names=label_names))
        
        print("")
        print('Training resuts') 
        tr_loss = tr_loss/(step+1)
        print("Train loss in epoch %d: %0.3f"%(epoch,tr_loss))
        print('classifiation report')
        print(classification_report(train_predict, train_labels, target_names=label_names))
        print('Accuracy:')
        matrix = confusion_matrix(train_labels, train_predict)
        print(matrix.diagonal()/matrix.sum(axis=1))

        
        '''
        Validation
        '''
        model.eval()
        valid_loss = 0
        valid_pred = []
        valid_labels = []
        with torch.no_grad():
            for step, (turns, labels, prompts) in tqdm(enumerate(val_loader)):
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
                
                
                valid_loss+=loss.item()
                
                # calculate preds
                if args.model_name == 'RoBERTa_entail':
                    valid_pred.extend(np.argmax(output.cpu().detach().numpy(),axis=-1))
                else:
                    valid_pred.extend(np.argmax(output[0].cpu().detach().numpy(),axis=-1))
                
                valid_labels.extend(_labels.cpu().detach().numpy())
                
                #if (step + 1) % args.statistic_step == 0:
                #    print('classifiation report')
                #    print(classification_report(valid_pred, valid_labels, target_names=label_names))
        
        print("")
        print('Validation resuts') 
        valid_loss = valid_loss/(step+1)
        print("Val loss in epoch %d: %0.3f"%(epoch,valid_loss))
        print('classifiation report')
        print(classification_report(valid_pred, valid_labels, target_names=label_names))
        print('Accuracy:')
        matrix = confusion_matrix(valid_labels, valid_pred)
        print(matrix.diagonal()/matrix.sum(axis=1))

        
        
        df.loc[epoch] = pd.Series({'Epoch':int(epoch), 'Tr Loss':round(tr_loss,4), 'Val Loss':round(valid_loss,4)})

        if valid_loss < best_loss:
            trigger_times = 0
            best_loss = valid_loss
            print('Found better model')
            print("Saving model to %s"%os.path.join(args.checkpoint_path,'_'.join([str(i) for i in model_bin])))
            model_to_save = model
            torch.save(model_to_save, os.path.join(args.checkpoint_path,'_'.join([str(i) for i in model_bin])))
        else:
            trigger_times += 1
            if trigger_times>= args.patience:
                df.to_csv(os.path.join(args.checkpoint_path,'_'.join([str(i) for i in model_bin]))+'.csv')
                print('Early Stopping!')
                break
    print("")
    df.to_csv(os.path.join(args.checkpoint_path,'_'.join([str(i) for i in model_bin]))+'.csv')
    print("Training complete!")  
            
        
        
    
    
if __name__ == "__main__":
    args = parse_arguments()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    USE_CUDA = torch.cuda.is_available()
    train(args)
    