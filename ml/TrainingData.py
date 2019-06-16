# -*- coding: utf-8 -*-

import numpy as np
import random

class TrainingData:
    def __init__(self, training_lists=[], batch_size=256, epochs=8):
        self.all_lists=training_lists
        self.sub_list=[]
        self.curr=0
        self.curr_epoch=0
        self.curr_data=0
        self.batch_size=batch_size
        self.epochs=epochs

    def __iter__(self):
        return self.gen_samples()

    def gen_samples(self):
        self.curr_data=0
        for list in self.all_lists:
            self.list=list
            self.curr_epoch=0
            self.len=len(self.list)
            while self.curr_epoch<self.epochs:
                self.curr=0
                random.shuffle(self.list)
                while self.curr<self.len:
                    from_idx=self.curr
                    self.curr+=min(self.batch_size-len(self.sub_list), self.len-self.curr)
                    self.sub_list=self.sub_list+self.list[from_idx:self.curr]
                    if len(self.sub_list)==self.batch_size:
                        yield self.sub_list
                        self.sub_list=[]
                self.curr_epoch+=1
            self.curr_data+=1
        if len(self.sub_list)>0:
            yield self.sub_list
