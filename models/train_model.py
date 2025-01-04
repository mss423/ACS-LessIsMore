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

def run_bert_train(data_train, data_test, num_labels, epochs=3):
    # data_train = load_data(path_train)
    # data_test = load_data(path_test)
    print(data_train.sample(10))

    args = ClassificationArgs(num_train_epochs=epochs, overwrite_output_dir=True)
    model = ClassificationModel(
        "bert", "bert-base-cased", num_labels=num_labels, args=args
    )
    model.train_model(data_train)
    result, model_outputs, wrong_predictions = model.eval_model(data_test)

    pred = model_outputs.argmax(-1).tolist()
    gold = data_test["labels"].tolist()
    print(classification_report(gold, pred))

def train_bert(train_df, dev_df, test_df, num_labels,
         max_seq_len=50,
         epochs=3,
         batch_size=32,
         lr=2e-05,
         patience=1,
         max_grad_norm=10.0,
         seed=0,
         shuffle=True):
    """
    Parameters
    ----------
    train_df : pandas dataframe of train set.
    dev_df : pandas dataframe of dev set.
    test_df : pandas dataframe of test set.
    max_seq_len: the max truncated length.
    epochs : the default is 3.
    batch_size : the default is 32.
    lr : learning rate, the default is 2e-05.
    patience : the default is 1.
    max_grad_norm : the default is 10.0.
    if_save_model: if save the trained model to the target dir.
    checkpoint : the default is None.
    seed: int for random seed

    """

    bertmodel = BertModel(num_labels = num_labels, requires_grad = True)
    tokenizer = bertmodel.tokenizer
    
    print(20 * "=", " Preparing for training ", 20 * "=")
        
    # -------------------- Data loading --------------------------------------#
    
    
    train_data = DataPrecessForSentence(tokenizer, train_df, max_seq_len = max_seq_len)
    dev_data = DataPrecessForSentence(tokenizer,dev_df, max_seq_len = max_seq_len)

    if shuffle:
        print("\t* Loading training data...")
        train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size, worker_init_fn = lambda id: np.random.seed(seed + epoch))
        print("\t* Loading validation data...")
        dev_loader   = DataLoader(dev_data, shuffle=shuffle, batch_size=batch_size, worker_init_fn = lambda id: np.random.seed(seed + epoch))
    else:
        print("\t* Loading training data...")
        train_loader = DataLoader(train_data, shuffle=shuffle, sampler=SequentialSampler(train_data), 
            batch_size=batch_size, worker_init_fn = lambda id: np.random.seed(seed + epoch))
        print("\t* Loading validation data...")
        dev_loader = DataLoader(dev_data, shuffle=shuffle, sampler=SequentialSampler(dev_data), 
            batch_size=batch_size, worker_init_fn = lambda id: np.random.seed(seed + epoch))

    print("\t* Loading test data...")
    test_data = DataPrecessForSentence(tokenizer, test_df, max_seq_len = max_seq_len) 
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    
    # -------------------- Model definition ------------------- --------------#
    
    print("\t* Building model...")
    device = torch.device("cuda")
    model = bertmodel.to(device)
    
    # -------------------- Preparation for training  -------------------------#
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {
                    'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    'weight_decay':0.01
            },
            {
                    'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    'weight_decay':0.0
            }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    ## Implement of warm up
    # total_steps = len(train_loader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)
    
    # When the monitored value is not improving, the network performance could be improved by reducing the learning rate.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    valid_aucs = []
        
     # Compute loss and accuracy before starting (or resuming) training.
    print("REACHED")
    _, valid_loss, valid_accuracy, auc, _, = validate(model, dev_loader)
    print("\n* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss, (valid_accuracy*100), auc))
    
    # -------------------- Training epochs -----------------------------------#
    
    print("\n", 20 * "=", "Training bert model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)  
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss, (epoch_accuracy*100)))
        
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy , epoch_auc, _, = validate(model, dev_loader)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_accuracy)
        valid_aucs.append(epoch_auc)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100), epoch_auc))
        
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        ## scheduler.step()
        
        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            
            # run model on test set and save the prediction result to csv
            print("* Test for epoch {}:".format(epoch))
            _, _, test_accuracy, _, all_prob = validate(model, test_loader)
            print("Test accuracy: {:.4f}%\n".format(test_accuracy))
            test_prediction = pd.DataFrame({'prob_1':all_prob})
            test_prediction['prob_0'] = 1-test_prediction['prob_1']
            test_prediction['prediction'] = test_prediction.apply(lambda x: 0 if (x['prob_0'] > x['prob_1']) else 1, axis=1)
            test_prediction = test_prediction[['prob_0', 'prob_1', 'prediction']]
             
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    os.system('clear')
    return test_prediction


def Metric(y_true, y_pred):
    """
    compute and show the classification result
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='macro')
    target_names = ['class_0', 'class_1']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3)

    print('Accuracy: {:.1%}\nPrecision: {:.1%}\nRecall: {:.1%}\nF1: {:.1%}'.format(accuracy, macro_precision,
                                           macro_recall, weighted_f1))
    print("classification_report:\n")
    print(report)
    return accuracy, weighted_f1
  
  
def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.
    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.
    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, dataloader, optimizer, epoch_number, max_gradient_norm):
    """
    Train a model for one epoch on some input data with a given optimizer and
    criterion.
    Args:
        model: A torch module that must be trained on some input data.
        dataloader: A DataLoader object to iterate over the training data.
        optimizer: A torch optimizer to use for training on the input model.
        epoch_number: The number of the epoch for which training is performed.
        max_gradient_norm: Max. norm for gradient norm clipping.
    Returns:
        epoch_time: The total time necessary to train the epoch.
        epoch_loss: The training loss computed for the epoch.
        epoch_accuracy: The accuracy computed for the epoch.
    """
    # Switch the model to train mode.
    model.train()
    device = model.device
    epoch_start = time.time()
    batch_time_avg = 0.0
    running_loss = 0.0
    correct_preds = 0
    tqdm_batch_iterator = tqdm(dataloader)
    for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(tqdm_batch_iterator):
        batch_start = time.time()
        # Move input and output data to the GPU if it is used.
        seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        loss, logits, probabilities = model(seqs, masks, segments, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        batch_time_avg += time.time() - batch_start
        running_loss += loss.item()
        correct_preds += correct_predictions(probabilities, labels)
        description = "Avg. batch proc. time: {:.4f}s, loss: {:.4f}"\
                      .format(batch_time_avg/(batch_index+1), running_loss/(batch_index+1))
        tqdm_batch_iterator.set_description(description)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = correct_preds / len(dataloader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, dataloader):
    """
    Compute the loss and accuracy of a model on some validation dataset.
    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
        roc_auc_score(all_labels, all_prob): The auc computed on the entire validation set.
        all_prob: The probability of classification as label 1 on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            # Move input and output data to the GPU if one is used.
            seqs = batch_seqs.to(device)
            masks = batch_seq_masks.to(device)
            segments = batch_seq_segments.to(device)
            labels = batch_labels.to(device)
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            running_loss += loss.item()
            running_accuracy += correct_predictions(probabilities, labels)
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(batch_labels)
    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader.dataset))

    # DEBUG -- handling multiclass predictions
    if len(np.unique(all_labels)) > 2:
        roc_auc = roc_auc_score(all_labels, all_prob, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(all_labels, all_prob)

    return epoch_time, epoch_loss, epoch_accuracy, roc_auc, all_prob


def test(model, dataloader):
    """
    Test the accuracy of a model on some labelled test dataset.
    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.
    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
        all_prob: The probability of classification as label 1 on the entire validation set.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device
    time_start = time.time()
    batch_time = 0.0
    accuracy = 0.0
    all_prob = []
    all_labels = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in dataloader:
            batch_start = time.time()
            # Move input and output data to the GPU if one is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(device), batch_labels.to(device)
            _, _, probabilities = model(seqs, masks, segments, labels)
            accuracy += correct_predictions(probabilities, labels)
            batch_time += time.time() - batch_start
            all_prob.extend(probabilities[:,1].cpu().numpy())
            all_labels.extend(batch_labels)
    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    accuracy /= (len(dataloader.dataset))

    return batch_time, total_time, accuracy, all_prob