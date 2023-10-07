import os
import json
import numpy as np
import re
from sklearn.model_selection import train_test_split
import argparse
import pyvi
from pyvi import ViTokenizer, ViPosTagger

def word_segment(sentence):
    sentence = ViTokenizer.tokenize(sentence)
    return sentence


def split_sentences(text):
    sentences = []
    current_sentence = ''
    in_quotes = False

    # List of common honorifics and abbreviations
    honorifics_list = ["Mr.", "Ms.", "Mrs.", "Dr.", "Ph.D.", "PhD", "etc."]

    for i in range(len(text)):
        char = text[i]
        current_sentence += char

        if char == '"':
            in_quotes = not in_quotes  # Toggle in-quotes state
        elif char in ('.', '!', '?') and not in_quotes:
            # Check if the period might be an abbreviation or honorific
            cur_word = current_sentence.split()[-1] if len(current_sentence.split()) > 1 else ""
            next_char = text[i + 1] if i < len(text) - 1 else ""

            if (
                cur_word in honorifics_list
                or re.match(r'^[A-Z][a-z]*\.$', cur_word) # check abbreviation (ex: Hubert S. Howe) 
                or (re.match(r'^[A-Z][a-z]*$', next_char)) # check next char is begin with an upcase
                #and not re.search(r'(?<=\.)[A-Z]', next_char)) # check there is no upcase behind a dot
                or next_char.isdigit()
            ):
                continue  # Skip sentence split if it meets the conditions

            sentences.append(current_sentence.strip())
            current_sentence = ''

    if current_sentence:
        sentences.append(current_sentence.strip())

    return sentences

def preprocess(doc):
    doc = re.sub("''", '"', doc)
    doc = re.sub("“", '"', doc)
    return doc

def split_context(context):
    sentences = []
    context = preprocess(context)
    paragraphs = context.split("\n\n")
    for parag in paragraphs:
        sentences.extend(split_sentences(parag))
    sentences = [sent for sent in sentences if sent != "."]

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
            value['claim'] = word_segment(value['claim'])
            context = value['context']
            splitted_context = split_context(context)
            segmented_context = [word_segment(sent) for sent in splitted_context]
            value['context'] = segmented_context
            if value['evidence']:
                value['evidence'] = word_segment(value['evidence'])
            data.append(value)
        return data
    
    def only_have_evidence_data(self, data):
        new_data = []
        for line in data:
            if line['evidence']:
                new_data.append(line)
        return new_data




if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, help="Name of input file")
    parser.add_argument('--outfile_train', type=str, help="Name of output train file")
    #parser.add_argument('--outfile_test', type=str, help="Name of the output test file")
    parser.add_argument('--outfile_valid', type=str, help="Name of the output valid file")
    args = parser.parse_args()

    gen_data = GenerateDataPair()
    data = gen_data.read_file(args.infile)
    data = gen_data.only_have_evidence_data(data)

    pairs = list()
    for line in data:
        claim = re.sub(r"[\.]", " ", line["claim"])
        claim = " ".join(claim.strip().split())
        evidence = re.sub(r"[\.]", " ", line["evidence"])
        evidence = " ".join(evidence.strip().split())
        context = line["context"]
        context = [re.sub(r"[\.]", " ", sent) for sent in context] 
        for sentence in context:
            sentence = " ".join(sentence.strip().split())
            if sentence != evidence:
                set_sent = [claim, evidence, sentence]
                print(len(set_sent), set_sent)
                pairs.append(set_sent)
     
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