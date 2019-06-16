# -*- coding: utf-8 -*-

from . import text_features

def sessions(input_file_path, sess_filter=lambda sess : True, qid_idx=1):
    qid='__not a qid'
    sess=[]
    with open(input_file_path) as f:
        for line in f:
            items=line.split('\t')
            if qid!=items[qid_idx]:
                if len(sess)>0 and sess_filter(sess):
                    yield sess
                sess=[]
                qid=items[qid_idx]
            sess.append(line)

def to_pairs(sess, mapping_dict, fea_size, shrink=True, feature_idx=2, label_idx=0):
    pos=[]
    neg=[]
    for line in sess:
        label, fea_list=text_features.mapping_line(line, mapping_dict, fea_size, shrink, feature_idx, label_idx)
        if label>0:
            pos.append(fea_list)
        else:
            neg.append(fea_list)
    sess_pairs=[]
    for one_pos in pos:
        for one_neg in neg:
            sess_pairs.append((one_neg, one_pos))
    return sess_pairs

def coverage(input_file_path, top_n_dist=10, to_n_contained=2, qid_idx=1, label_idx=0, score_idx=3):
    covered=[0]*top_n_dist
    class EstimateRecord:
        pass
    contained_pos_sess=0
    contained_in_top_pos_sess=0
    for sess in sessions(input_file_path, qid_idx=qid_idx):
        record_list=[]
        for line in sess:
            items=line.split('\t')
            record=EstimateRecord()
            record.label=float(items[label_idx])
            record.score=float(items[score_idx])
            record_list.append(record)
        record_list.sort(key=lambda x : -x.score)
        i=0
        contained_pos=False
        in_top=False
        for record in record_list:
            if record.label==1.0:
                contained_pos=True
                covered[min(i, top_n_dist-1)]+=1
                if i<to_n_contained:
                    in_top=True
            i+=1
        if contained_pos:
            contained_pos_sess+=1
        if in_top:
            contained_in_top_pos_sess+=1

    return covered, contained_in_top_pos_sess/contained_pos_sess
