# -*- coding: utf-8 -*-

import numpy as np

class FeatureStatistics:
    def __init__(self, idx=-1):
        self.idx=idx
        self.min=99999999
        self.max=-99999999
        self.count=0
        self.positive=0
        self.negtive=0

    def update(self, value, is_positive):
        if value<self.min:
            self.min=value
        if value>self.max:
            self.max=value
        self.count+=1
        if is_positive:
            self.positive+=1
        else:
            self.negtive+=1

    def from_line(self, line):
        idx, min, max, count, positive, negtive=line.split('\t')
        self.idx=int(idx)
        self.min=float(min)
        self.max=float(max)
        self.count=int(count)
        self.positive=int(positive)
        self.negtive=int(negtive)

    def read_mapping(self, line):
        idx, k, b=line.split('\t')
        self.idx=int(idx)
        self.k=float(k)
        self.b=float(b)

def calc_feature_statistics(input_file_path, feature_idx=2, label_idx=0):
    d={}
    with open(input_file_path) as f:
        for line in f:
            items=line.split('\t')
            label=items[label_idx]
            for fea in items[feature_idx].split(' '):
                idx, value=fea.split(':', 1)
                idx=int(idx)
                value=float(value)
                if idx not in d:
                    d[idx]=FeatureStatistics(idx)
                d[idx].update(value, label=='1')

    return d, max([x for x in d])

def save_feature_statistics(fea_dict, output_file_path):
    fea_list=[fs for fs in fea_dict.values()]
    fea_list.sort(key=lambda x : x.idx)
    with open(output_file_path, 'w') as f:
        for fs in fea_list:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fs.idx, fs.min, fs.max, fs.count, fs.positive, fs.negtive))

def load_feature_statistics(file_path):
    d={}
    with open(file_path) as f:
        for line in f:
            fs=FeatureStatistics()
            fs.from_line(line)
            d[fs.idx]=fs
    return d, max([x for x in d])

def set_mapping(feature_dict, lower, upper, scale=1, zero_original_lower=True):
    for fea in feature_dict.values():
        original_lower=(fea.min if fea.min<0 else 0) if zero_original_lower else fea.min
        fea.k=(upper-lower)*scale/(fea.max-original_lower)
        fea.b=-fea.k*original_lower+lower+(upper-lower)*(1-scale)/2
    fea_list=[fs for fs in feature_dict.values()]
    fea_list.sort(key=lambda x : x.idx)
    for i in range(0, len(fea_list)):
        fea_list[i].compact_idx=i

def save_mapping(fea_dict, output_file_path):
    fea_list=[fs for fs in fea_dict.values()]
    fea_list.sort(key=lambda x : x.idx)
    with open(output_file_path, 'w') as f:
        for fs in fea_list:
            f.write('{0}\t{1:.10f}\t{2:.10f}\n'.format(fs.idx, fs.k, fs.b))

def load_mapping(file_path):
    d={}
    with open(file_path) as f:
        idx=0
        for line in f:
            fs=FeatureStatistics()
            fs.read_mapping(line)
            fs.compact_idx=idx
            idx+=1
            d[fs.idx]=fs
    return d, max([x for x in d])

def mapping_line(line, mapping_dict, fea_size, shrink=False, feature_idx=2, label_idx=0):
    fea_list=[0]*fea_size
    items=line.split('\t')
    label=float(items[label_idx])
    for fea in items[feature_idx].split(' '):
        idx, value=fea.split(':', 1)
        idx=int(idx)
        value=float(value)
        fea_mapping=mapping_dict[idx]
        if fea_mapping is not None:
            if shrink:
                idx=fea_mapping.compact_idx
            fea_list[idx]=fea_mapping.k*value+fea_mapping.b
    return label, fea_list

def mapping(input_file_path, mapping_dict, max_idx, shrink=False, feature_idx=2, label_idx=0):
    size=len(mapping_dict) if shrink else max_idx+1
    with open(input_file_path) as f:
        for line in f:
            label, fea_list=mapping_line(line, mapping_dict, size, shrink, feature_idx, label_idx)
            yield label, fea_list

def to_binary_feature_file(feature_list, output_file):
    with open(output_file, 'wb') as f:
        for label, fea in feature_list:
            data=np.array([label]+fea)
            data.astype('float32').tofile(f)

def to_text_feature_file(feature_list, output_file):
    with open(output_file, 'w') as f:
        for label, fea in feature_list:
            s=''
            for idx, value in enumerate(fea):
                if value!=0:
                    s+='{}:{} '.format(idx, value)
            f.write('{}\t{}\n'.format(label, s.strip()))

if __name__=='__main__':
    import sys

    if sys.argv[1]=='calc_fea':
        #./run <flag> <sample_file> <feature_statistics_file>
        fea_dict, _=calc_feature_statistics(sys.argv[2])
        save_feature_statistics(fea_dict, sys.argv[3])
    elif sys.argv[1]=='calc_fea:to_bin':
        #./run <flag> <sample_file> <feature_statistics_file> <mapping_file> <binary_sample_output_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=calc_feature_statistics(sys.argv[2])
        save_feature_statistics(fea_dict, sys.argv[3])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[6]) if arg_len>6 else 0, float(sys.argv[7]) if arg_len>7 else 1, float(sys.argv[8]) if arg_len>8 else 0.8)
        save_mapping(fea_dict, sys.argv[4])
        to_binary_feature_file(mapping(sys.argv[2], fea_dict, max_idx, True), sys.argv[5])
    elif sys.argv[1]=='calc_fea:calc_map':
        #./run <flag> <sample_file> <feature_statistics_file> <mapping_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=calc_feature_statistics(sys.argv[2])
        save_feature_statistics(fea_dict, sys.argv[3])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[5]) if arg_len>5 else 0, float(sys.argv[6]) if arg_len>6 else 1, float(sys.argv[7]) if arg_len>7 else 0.8)
        save_mapping(fea_dict, sys.argv[4])
    elif sys.argv[1]=='calc_map':
        #./run <flag> <feature_statistics_file> <mapping_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=load_feature_statistics(sys.argv[2])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[4]) if arg_len>4 else 0, float(sys.argv[5]) if arg_len>5 else 1, float(sys.argv[6]) if arg_len>6 else 0.8)
        save_mapping(fea_dict, sys.argv[3])
    elif sys.argv[1]=='calc_map:to_bin':
        #./run <flag> <sample_file> <feature_statistics_file> <mapping_file> <binary_sample_output_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=load_feature_statistics(sys.argv[3])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[6]) if arg_len>6 else 0, float(sys.argv[7]) if arg_len>7 else 1, float(sys.argv[8]) if arg_len>8 else 0.8)
        save_mapping(fea_dict, sys.argv[4])
        to_binary_feature_file(mapping(sys.argv[2], fea_dict, max_idx, True), sys.argv[5])
    elif sys.argv[1]=='to_bin':
        #./run <flag> <sample_file> <mapping_file> <binary_sample_output_file>
        mapping_dict, max_idx=load_mapping(sys.argv[3])
        to_binary_feature_file(mapping(sys.argv[2], mapping_dict, max_idx, True), sys.argv[4])
    elif sys.argv[1]=='calc_fea:to_text':
        #./run <flag> <sample_file> <feature_statistics_file> <mapping_file> <text_sample_output_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=calc_feature_statistics(sys.argv[2])
        save_feature_statistics(fea_dict, sys.argv[3])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[6]) if arg_len>6 else 0, float(sys.argv[7]) if arg_len>7 else 1, float(sys.argv[8]) if arg_len>8 else 0.8)
        save_mapping(fea_dict, sys.argv[4])
        to_text_feature_file(mapping(sys.argv[2], fea_dict, max_idx), sys.argv[5])
    elif sys.argv[1]=='calc_map:to_text':
        #./run <flag> <sample_file> <feature_statistics_file> <mapping_file> <text_sample_output_file> (<lower> <upper> <scale>)
        fea_dict, max_idx=load_feature_statistics(sys.argv[3])
        arg_len=len(sys.argv)
        set_mapping(fea_dict, float(sys.argv[6]) if arg_len>6 else 0, float(sys.argv[7]) if arg_len>7 else 1, float(sys.argv[8]) if arg_len>8 else 0.8)
        save_mapping(fea_dict, sys.argv[4])
        to_text_feature_file(mapping(sys.argv[2], fea_dict, max_idx), sys.argv[5])
    elif sys.argv[1]=='to_text':
        #./run <flag> <sample_file> <mapping_file> <text_sample_output_file>
        mapping_dict, max_idx=load_mapping(sys.argv[3])
        to_text_feature_file(mapping(sys.argv[2], mapping_dict, max_idx), sys.argv[4])
    else:
        print('error')
