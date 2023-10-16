import random, os
from tqdm import tqdm
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModel

from models import inference_model
from data_loader_v2 import DataLoader, DataLoaderTest
from models_v2 import BertForSequenceEncoder, TripletLoss
from torch.nn import NLLLoss
import logging
import json
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    if 1.0 - x >= 0.0:
        return 1.0 - x
    return 0.0

def correct_prediction(anchor, pos, neg):
    correct = 0.0
    anchor = anchor.detach().numpy()
    pos = pos.detach().numpy()
    neg = neg.detach().numpy
    pos_distance = cosine_similarity(anchor, pos)
    neg_distance = cosine_similarity(anchor, neg)
    for step in range(len(anchor)):
        if pos_distance[step] > neg_distance[step]:
            correct += 1
    return correct

def eval_model(model, validset_reader):
    model.eval()
    correct_pred = 0.0
    for inp_tensor_anchor, msk_tensor_anchor, inp_tensor_pos, msk_tensor_pos, inp_tensor_neg, msk_tensor_neg in validset_reader:
        anchor = model(inp_tensor_anchor, msk_tensor_anchor)
        pos = model(inp_tensor_pos, msk_tensor_pos)
        neg = model(inp_tensor_neg, msk_tensor_neg)
        correct_pred += correct_prediction(anchor, pos, neg)
    dev_accuracy = correct_pred / validset_reader.total_num
    return dev_accuracy



def train_model(model, args, trainset_reader, validset_reader, criterion):
    save_path = args.outdir + '/model'
    best_acc = 0.0
    running_loss = 0.0
    t_total = int(
        trainset_reader.total_num / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    
    optimizer = Adam(model.parameters(), args.learning_rate)                      
    
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step : warmup_linear(step / t_total))
    
    global_step = 0
    for epoch in range(int(args.num_train_epochs)):
        print("----------EPOCH {}----------".format(epoch))
        optimizer.zero_grad()
        for inp_tensor_anchor, msk_tensor_anchor, inp_tensor_pos, msk_tensor_pos, inp_tensor_neg, msk_tensor_neg in tqdm(trainset_reader):
            model.train()
            anchor = model(inp_tensor_anchor, msk_tensor_anchor)
            pos = model(inp_tensor_pos, msk_tensor_pos)
            neg = model(inp_tensor_neg, msk_tensor_neg)
            loss = criterion(anchor, pos, neg)
            running_loss += loss.item()
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                logger.info('Epoch: {0}, Step: {1}, Loss: {2}'.format(epoch, global_step, (running_loss / global_step)))
            
        # Eval after training
        logger.info('Start epoch {} eval!'.format(epoch))
        print('Start epoch {} eval!'.format(epoch))
        eval_acc = eval_model(model, validset_reader)
        logger.info('Dev acc: {0}'.format(eval_acc))
        print('Dev acc: {0}'.format(eval_acc))
        if eval_acc >= best_acc:
            best_acc = eval_acc
            torch.save({'epoch': epoch,
                        'model': model.state_dict()}, save_path + ".best.pt")
            logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--train_path', help='train path')
    parser.add_argument('--valid_path', help='valid path')
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--bert_hidden_dim", default=768, type=int, help="Total batch size for training.")
    parser.add_argument("--total_num_valid", default=100000, type=int, help="Total num of validation.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument('--outdir', required=True, help='path to output directory')
    parser.add_argument("--pool", type=str, default="att", help='Aggregating method: top, max, mean, concat, att, sum')
    parser.add_argument("--layer", type=int, default=1, help='Graph Layer.')
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--evi_num", type=int, default=5, help='Evidence num.')
    parser.add_argument("--threshold", type=float, default=0.0, help='Threshold.')
    parser.add_argument("--max_len", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--eval_step", default=2000, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument('--bert_pretrain', default="bert-base-uncased")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% "
                             "of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=4,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float, help="Hidden dropout probability.")
    parser.add_argument("--checkpoint", type=str, help="Pretrained model path")


    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    handlers = [logging.FileHandler(os.path.abspath(args.outdir) + '/train_log.txt'), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    
    logger.info('Start training!')

    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain)
    logger.info("loading training set")
    trainset_reader = DataLoader(args.train_path, tokenizer, args, batch_size=args.train_batch_size)
    logger.info("loading validation set")
    # Use this line to set total number samples as total_num_valid
    #validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size, test=True)
    # Use this line to set total number samples as number of lines read in file
    validset_reader = DataLoader(args.valid_path, tokenizer, args, batch_size=args.valid_batch_size, test=False)

    logger.info('initializing estimator model')
    bert_model = BertForSequenceEncoder(args.bert_pretrain, args)
    bert_model = bert_model.cuda()
    model = inference_model(bert_model, args)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
    model = model.cuda()
    criterion = TripletLoss(margin=1.0)
    train_model(model, args, trainset_reader, validset_reader, criterion)