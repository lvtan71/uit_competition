import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU
from bert_model import BertForSequenceEncoder

from torch.nn import BatchNorm1d, Linear, ReLU
from torch.autograd import Variable
import numpy as np
from transformers import AutoModel


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # Euclidean distance between anchor and positive
        distance_negative = (anchor - negative).pow(2).sum(1)  # Euclidean distance between anchor and negative
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class BertForSequenceEncoder(nn.Module):
    def __init__(self, model_name, args):
        super(BertForSequenceEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask)
            output = outputs.last_hidden_state
            #output = self.dropout(output)
            pooled_output = outputs.pooler_output
            #pooled_output = self.dropout(pooled_output)
        return output, pooled_output


class inference_model(nn.Module):
    def __init__(self, bert_model, args):
        super(inference_model, self).__init__()
        self.bert_hidden_dim = args.bert_hidden_dim
        self.dropout = nn.Dropout(args.dropout)
        self.max_len = args.max_len
        self.num_labels = args.num_labels
        self.pred_model = bert_model


    def forward(self, inp_tensor, msk_tensor):
        _, inputs = self.pred_model(inp_tensor, msk_tensor)
        #inputs = self.dropout(inputs) 
        return inputs