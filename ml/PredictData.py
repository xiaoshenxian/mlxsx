# -*- coding: utf-8 -*-

import numpy as np

class PredictData:
    def __init__(self, predict_list, batch_size=256):
        self.all_lists=predict_list
        self.sub_list=[]
        self.curr=0
        self.curr_data=0
        self.batch_size=batch_size

    def __iter__(self):
        return self.gen_samples()
 
    def gen_samples(self):
        self.curr_data=-1
        for list in self.all_lists:
            self.curr_data+=1
            self.list=list
            self.len=len(self.list)
            self.curr=0
            while self.curr<self.len:
                from_idx=self.curr
                self.curr+=min(self.batch_size-len(self.sub_list), self.len-self.curr)
                self.sub_list=self.sub_list+self.list[from_idx:self.curr]
                if len(self.sub_list)==self.batch_size:
                    yield self.sub_list
                    self.sub_list=[]
        if len(self.sub_list)>0:
            yield self.sub_list
