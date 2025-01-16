import os
import pandas as pd
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from sys import platform

from data_precess import DataPrecessForSentence
from models import BertModel

from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs
from simpletransformers.ner import NERArgs, NERModel

import numpy as np

import torch
import torch.nn as nn
import time
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report)

def train_on_data(data_train, data_test, num_labels, dataset, epochs=3, seed=0):
    if dataset in ['sst2','fewrel']:
        return run_bert_train(data_train, data_test, num_labels, seed=seed)
    elif dataset == 'aste':
        # todo
        print("implement!")
    elif dataset == 'crossner':
        # todo
        print("implement!")
    else:
        raise ValueError('Invalid dataset name passed!')

def run_bert_train(data_train, data_test, num_labels, epochs=3, seed=0):
    # print(data_train.sample(10))

    args = ClassificationArgs(num_train_epochs=epochs, overwrite_output_dir=True, manual_seed=seed)
    model = ClassificationModel(
        "bert", "bert-base-cased", num_labels=num_labels, args=args
    )
    model.train_model(data_train)
    result, model_outputs, wrong_predictions = model.eval_model(data_test)

    pred = model_outputs.argmax(-1).tolist()
    gold = data_test["label"].tolist()
    return classification_report(gold, pred, output_dict=True, zero_division=0.0), accuracy_score(gold, pred)

def run_ner_train(data_train, data_test, epochs=3, seed=0):
    custom_labels = ["O", "B-algorithm", "I-algorithm", "B-conference", "I-conference", \
        "B-country", "I-country", "B-field", "I-field", "B-metrics", "I-metrics", \
        "B-misc", "I-misc", "B-organisation", "I-organisation", "B-person", "I-person", \
        "B-product", "I-product", "B-programming_language", "I-programming_language", \
        "B-researcher", "I-researcher", "B-task", "I-task", "B-university", "I-university"]

    args = NERArgs(labels_list=custom_labels, num_train_epochs=epochs, overwrite_output_dir=True, manual_seed=seed)
    args.labels_list = custom_labels
    model = NERModel("bert", "bert-base-cased", args=args)
    model.train_model(data_train)
    result, model_outputs, wrong_preds = model.eval_model(data_test)

    pred = model_outputs.argmax(-1).tolist()
    gold = data_test["label"].tolist()
    return classification_report(gold, pred, output_dict=True, zero_division=0.0), accuracy_score(gold, pred)

