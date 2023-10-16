import torch
import numpy as np
import json
from torch.autograd import Variable
import re
from pyvi import ViTokenizer



def split_sentences(text):
    sentences = []
    current_sentence = ''
    in_quotes = False

    # List of common honorifics and abbreviations
    honorifics_list = ["Mr.", "Ms.", "Mrs.", "Dr.", "Ph.D.", "PhD", "etc.",
                       "TS.", "PSG.", "Th.S.", "ThS.", "GS.", "BS.", "KS.",
                       "approx.", "appt.", "apt.", "dept.", "est.", "min.",
                       "TP.", "Tp.", "Ô.B."]

    for i in range(len(text)):
        char = text[i]
        current_sentence += char

        if char == '"':
            in_quotes = not in_quotes  # Toggle in-quotes state
        elif char in ('.', '!', '?') and not in_quotes:
            # Check if the period might be an abbreviation or honorific
            cur_word = current_sentence.split()[-1] if len(current_sentence.split()) >= 1 else ""
            next_char = text[i + 1] if i < len(text) - 1 else ""

            #print(cur_word, "-", next_char)
            if (
                len([word for word in honorifics_list if word in cur_word])
                or re.match(r'^[A-Z]\.$', cur_word) # check abbreviation (ex: Hubert S. Howe)
                or (re.match(r'^[A-Za-z]*$', next_char)) # check next char is begin with an upcase
                #and not re.search(r'(?<=\.)[A-Z]', next_char)) # check there is no upcase behind a dot
                or next_char == "."
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


def tok2int_sent(sentence, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    inputs = tokenizer(sentence,
                       padding='max_length',
                       max_length=max_seq_length,
                       truncation=True,
                       return_tensors='pt',
                       add_special_tokens=True)
    input_ids = inputs["input_ids"][0].tolist()
    input_mask = inputs["attention_mask"][0].tolist()
    
    return input_ids, input_mask



def tok2int_list(src_list, tokenizer, max_seq_length):
    inp_padding = list()
    msk_padding = list()
    for sent in src_list:
        input_ids, input_mask = tok2int_sent(sent, tokenizer, max_seq_length)
        inp_padding.append(input_ids)
        msk_padding.append(input_mask)
        
    return inp_padding, msk_padding



class DataLoader(object):
    """For data iteration"""
    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.data_path = data_path
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_num = args.total_num_valid
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
            self.shuffle()
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0



    def process_sent(self, sentence):  
        sentence = re.sub("''", '"', sentence)
        sentence = re.sub("“", '"', sentence)
        sentence = re.sub(r"['\",\.\?:\-!]", "", sentence)
        return sentence
    


    def read_file(self, data_path):
        examples = list()
        with open(data_path) as fin:
            for line in fin:
                sublines = line.strip().split("\t")
                examples.append([self.process_sent(sublines[0]), self.process_sent(sublines[1]), self.process_sent(sublines[2])])
        return examples

    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        """Get the next batch"""
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1) * self.batch_size]
            anchor_inputs = list()
            pos_inputs = list()
            neg_inputs = list()
            for example in examples:
                anchor_inputs.append(example[0])
                pos_inputs.append(example[1])
                neg_inputs.append(example[2])
            inp_anchor, msk_anchor = tok2int_list(anchor_inputs, self.tokenizer, self.max_len)
            inp_pos, msk_pos = tok2int_list(pos_inputs, self.tokenizer, self.max_len)
            inp_neg, msk_neg = tok2int_list(neg_inputs, self.tokenizer, self.max_len)

            inp_tensor_anchor = Variable(torch.LongTensor(inp_anchor))
            msk_tensor_anchor = Variable(torch.LongTensor(msk_anchor))
            inp_tensor_pos = Variable(torch.LongTensor(inp_pos))
            msk_tensor_pos = Variable(torch.LongTensor(msk_pos))
            inp_tensor_neg = Variable(torch.LongTensor(inp_neg))
            msk_tensor_neg = Variable(torch.LongTensor(msk_neg))

                        
            if self.cuda:
                inp_tensor_anchor = inp_tensor_anchor.cuda()
                msk_tensor_anchor = msk_tensor_anchor.cuda()
                inp_tensor_pos = inp_tensor_pos.cuda()
                msk_tensor_pos = msk_tensor_pos.cuda()
                inp_tensor_neg = inp_tensor_neg.cuda()
                msk_tensor_neg = msk_tensor_neg.cuda()
            self.step += 1
            return inp_tensor_anchor, msk_tensor_anchor, inp_tensor_pos, msk_tensor_pos, inp_tensor_neg, msk_tensor_neg
        else:
            self.step = 0
            if not self.test:
                #examples = self.read_file(self.data_path)
                #self.examples = examples
                self.shuffle()
            raise StopIteration()
        

class DataLoaderTest(object):
    """For data iteration"""

    def __init__(self, data_path, tokenizer, args, cuda=True, batch_size=64):
        self.cuda = cuda

        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = args.max_len
        self.data_path = data_path
        inputs_anchor, inputs_sentence, ids, sentence_list, claim_list = self.read_file(data_path)
        self.inputs_anchor = inputs_anchor
        self.inputs_sentence = inputs_sentence
        self.ids = ids
        self.sentence_list = sentence_list
        self.claim_list = claim_list

        self.total_num = len(inputs_anchor)
        self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        self.step = 0


    def word_segment(self, sentence):
        sentence = ViTokenizer.tokenize(sentence)
        return sentence


    def process_sent(self, sentence):
        # sentence = re.sub(" \-LSB\-.*?\-RSB\-", "", sentence)
        # sentence = re.sub("\-LRB\- \-RRB\- ", "", sentence)
        # sentence = re.sub(" -LRB-", " ( ", sentence)
        # sentence = re.sub("-RRB-", " )", sentence)
        # sentence = re.sub("--", "-", sentence)
        # sentence = re.sub("``", '"', sentence)
        sentence = re.sub("''", '"', sentence)
        sentence = re.sub("“", '"', sentence)
        sentence = re.sub(r"['\",\.\?:\-!]", "", sentence)

        return sentence
    
    def read_file(self, data_path):
        inputs_anchor = list()
        inputs_sentence = list()
        ids = list()
        sentence_list = list()
        claim_list = list()
        with open(data_path, 'r', encoding='utf-8') as fin:
            data = json.load(fin)
        for key, value in data.items():
            claim = self.word_segment(value['claim'])
            id = key
            context = value['context']
            sentences = split_context(context)
            sentences = [self.word_segment(sent) for sent in sentences]
            for sentence in sentences:
                ids.append(id)
                inputs_anchor.append(self.process_sent(claim))
                inputs_sentence(self.process_sent(sentence))
                sentence_list.append(sentence)
                claim_list.append(claim)
        return inputs_anchor, inputs_sentence, ids, sentence_list, claim_list

    def __iter__(self):
        return self
    
    def __next__(self):
        return self.next()
    
    def __len__(self):
        return self._n_batch
    
    def next(self):
        """Get next batch"""
        if self.step < self.total_step:
            inputs = self.inputs[self.step * self.batch_size : (self.step+1) * self.batch_size]
            ids = self.ids[self.step * self.batch_size : (self.step+1) * self.batch_size]
            sentence_list = self.sentence_list[self.step * self.batch_size : (self.step+1) * self.batch_size]
            claim_list = self.claim_list[self.step * self.batch_size : (self.step+1) * self.batch_size]
            inp, msk = tok2int_list(inputs, self.tokenizer, self.max_len)
            inp_tensor_input = Variable(
                torch.LongTensor(inp)
            )
            msk_tensor_input = Variable(
                torch.LongTensor(msk)
            )

            if self.cuda:
                inp_tensor_input = inp_tensor_input.cuda()
                msk_tensor_input = msk_tensor_input.cuda()
            self.step += 1
            return inp_tensor_input, msk_tensor_input, ids, sentence_list, claim_list
        else:
            self.step = 0
            raise StopIteration()