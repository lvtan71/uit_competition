import os
import argparse
import re
import json
import numpy as np
from tqdm import tqdm

def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text


def accuracy(gt, pred):
    gt_evidence = gt["evidence"]
    pred_evidence = pred["evidence"]
    gt_evidence = preprocess_text(gt_evidence)
    pred_evidence = preprocess_text(pred_evidence)

    acc_1 = int(gt_evidence==pred_evidence)

    return acc_1



def evaluate(groundtruth_file, predict_file):
    with open(groundtruth_file, "r", encoding="utf-8") as f_groundtruth:
        gt_data = json.load(f_groundtruth)
    
    with open(predict_file, "r", encoding="utf-8") as f_predict:
        pred_data = json.load(f_predict)
    
    assert gt_data.keys() == pred_data.keys()

    ids = list(gt_data.keys())
    accs_1 = []
    for id in tqdm(ids):
        gt = gt_data[id]
        pred = pred_data[id]
        if gt["evidence"]:
            score = accuracy(gt, pred)
            accs_1.append(score)

    score = np.array(accs_1).mean()

    return score

if __name__=="__name__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", type=str, help="Name of output prediction file")
    parser.add_argument("--groundtruth_file", type=str, help="Name of groundtruth file")
    args = parser.parse_args()

    print("Accuracy: ", evaluate(args.groundtruth_file, args.predict_file))