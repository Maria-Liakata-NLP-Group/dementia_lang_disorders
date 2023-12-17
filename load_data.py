import os
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch

class Load_Data():
     def __init__(self, path: str, batch_size: int, flag:str):
         self.path = path
         self.batch_size = batch_size
         self.flag = flag
         
         self.dataset_file = open(self.path,'rb')
         self.dataset = pickle.load(self.dataset_file)
         self.dataset_file.close()
         
    
     def loader(self):
         #Load data
         
         
         
         
         #Load data on DataLoader
         if 'train' in self.flag:
             # Create the DataLoaders for our training and validation sets.
             # We'll take training samples in random order. 
             dataloader = DataLoader(
                 self.dataset,  # The training samples.
                 sampler = RandomSampler(self.dataset), # Select batches randomly
                 batch_size = self.batch_size # Trains with this batch size.
             )
         else:
             # For validation and test the order doesn't matter, so we'll just read them sequentially.
             dataloader = DataLoader(
                self.dataset, # The validation samples.
                sampler = SequentialSampler(self.dataset), # Pull out batches sequentially.
                batch_size = self.batch_size # Evaluate with this batch size.
            )
         
         #Returns input_ids, masks, token_type_ids, and labels    
         return dataloader 

