from torch.utils.data import Dataset
import random


class Dataload_prompt(Dataset):
    
    def __init__(self, df, df_dem):
        self.df = df
        self.df_dem = df_dem
        self.turn = df['Turn'].values
        
    def __getitem__(self, item):
        turn = self.turn[item]
        
            
        fluent = self.df_dem[(self.df_dem['type_ordinal']==0) & (self.df_dem['Turn']!=turn)].sample(1)
        fluent_turn = fluent['Turn'].to_list()[0]
        fluent_prompt = fluent['Type'].to_list()[0]
        
        anomia = self.df_dem[(self.df_dem['type_ordinal']==1) & (self.df_dem['Turn']!=turn)].sample(1)
        anomia_turn = anomia['Turn'].to_list()[0]
        anomia_prompt = anomia['Type'].to_list()[0]
            
        disf = self.df_dem[(self.df_dem['type_ordinal']==2) & (self.df_dem['Turn']!=turn)].sample(1)
        disf_turn = disf['Turn'].to_list()[0]
        disf_prompt = disf['Type'].to_list()[0]
            
        gram = self.df_dem[(self.df_dem['type_ordinal']==3) & (self.df_dem['Turn']!=turn)].sample(1)
        gram_turn = gram['Turn'].to_list()[0]
        gram_prompt = gram['Type'].to_list()[0]
            
        turn = turn + '. It is <mask>. ' + fluent_turn + '. It is ' + fluent_prompt + '. ' + anomia_turn + '. It is ' + anomia_prompt + '. ' + disf_turn + '. It is ' + disf_prompt + '. '  + gram_turn + '. It is ' + gram_prompt + '.'  
            
            
        return (turn)
        
    def __len__ (self):
        return len(self.turn)# -*- coding: utf-8 -*-