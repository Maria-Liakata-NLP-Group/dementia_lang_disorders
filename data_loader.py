from torch.utils.data import Dataset

class Dataload(Dataset):
    
    def __init__(self, df):
        self.df = df
        self.turn = df['Turn'].values
        self.label = df['type_ordinal'].values
        self.prompts = df['Type'].values
        self.index = df.index.values
        
    def __getitem__(self, item):
       
        turn = self.turn[item]
        label = self.label[item]
        prompts = self.prompts[item]
        
        return (turn, label, prompts)
        
    def __len__ (self):
        return len(self.turn)# -*- coding: utf-8 -*-