import os
import json
import numpy as np
import re
from sklearn.model_selection import train_test_split
import argparse
import py_vncorenlp

def word_segment(document):
    sentences = rdrsegmenter.word_segment(document)
    return sentences


class GenerateDataPair(object):
    def __init__(self, data_path=None, test_size=None, shuffle=None):
        self.data_path = data_path
        self.test_size = test_size
        self.shuffle = shuffle

    def read_file(self, data_path):
        data = list()
        with open(data_path, 'r', encoding='utf-8') as f:
            pure_data = json.load(f)
        for key, value in pure_data.items():
            value['key'] = key
            value['claim'] = word_segment(value['claim'])[0]
            value['context'] = word_segment(value['context'])
            if value['evidence']:
                value['evidence'] = word_segment(value['evidence'])[0]
            data.append(value)
        return data
    
    def only_have_evidence_data(self, data):
        new_data = []
        for line in data:
            if line["evidence"]:
                new_data.append(line)
        return new_data




if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, help="Name of input file")
    parser.add_argument('--outfile_train', type=str, help="Name of output train file")
    #parser.add_argument('--outfile_test', type=str, help="Name of the output test file")
    parser.add_argument('--outfile_valid', type=str, help="Name of the output valid file")
    parser.add_argument('--vncorenlp_dir', type=str, help="Path to VnCoreNLP")
    args = parser.parse_args()

    os.makedirs("./vncorenlp", exist_ok=True)
    py_vncorenlp.download_model(save_dir='./vncorenlp')
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=['wseg'], save_dir=args.vncorenlp_dir)

    gen_data = GenerateDataPair()
    data = gen_data.read_file(args.infile)
    data = gen_data.only_have_evidence_data(data)

    pairs = list()
    for line in data:
        claim = re.sub(r"[\.]", "", line["claim"])
        claim = " ".join(claim.strip().split())
        evidence = re.sub(r"[\.]", "", line["evidence"])
        evidence = " ".join(evidence.strip().split())
        context = line["context"]
        context = [re.sub(r"[\.]", "", sent) for sent in context]
        for sentence in context:
            sentence = " ".join(sentence.strip().split())
            if sentence != evidence:     
                pairs.append([claim, evidence, sentence])
     
    train_data, valid_data = train_test_split(pairs, test_size=0.2, shuffle=True)
    #valid_data, testdata = train_test_split(valid_data, test_size=0.5, shuffle=True)

    with open(args.outfile_train, 'w') as out:
        for pair in train_data: 
            try:
                out.write("\t".join(pair) + "\n")  
            except:
                continue

    # with open(args.outfile_test, 'w') as out:
    #     for pair in test_data:
    #         try:
    #             out.write("\t".join(pair) + "\n")
    #         except:
    #             continue

    with open(args.outfile_valid, 'w') as out:
        for pair in valid_data:
            try:
                out.write("\t".join(pair) + "\n")
            except:
                continue