#!/usr/bin/env python
# coding: utf-8
from preprocess import preprocess_dataset
from dataset import QuizBowlDataset
import json
from typing import List, Optional, Tuple
import os
import shutil
import random
import time
from util import get, get_tmp_filename
import numpy as np
import cloudpickle
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from allennlp.modules.elmo import Elmo, batch_to_ids
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Any, Dict, Optional
import abc
from collections import defaultdict
from urllib import request
import torch
from torch.autograd import Variable
import logging

QuestionText = str
Page = str
Evidence = Dict[str, Any]
TrainingData = Tuple[List[List[QuestionText]], List[Page], Optional[List[Evidence]]]

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_DIM = 1024
CUDA = torch.cuda.is_available()

def create_save_model(model):
    def save_model(path):
        torch.save(model.state_dict(), path)
    return save_model

class ElmoModel(nn.Module):
    def __init__(self, n_classes, dropout=.5):
        super().__init__()
        self.dropout = dropout
        # This turns off gradient updates for the elmo model, but still leaves scalar mixture
        # parameters as tunable, provided that references to the scalar mixtures are extracted
        # and plugged into the optimizer
        self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, 2, dropout=dropout, requires_grad=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * ELMO_DIM, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.Dropout(dropout)
        )

    def forward(self, questions, lengths):
        embeddings = self.elmo(questions)
        layer_0 = embeddings['elmo_representations'][0]
        layer_0 = layer_0.sum(1) / lengths
        layer_1 = embeddings['elmo_representations'][1]
        layer_1 = layer_1.sum(1) / lengths
        layer = torch.cat([layer_0, layer_1], 1)
        return self.classifier(layer)

def batchify(x_data, y_data, batch_size=10, shuffle=False):
    batches = []
    for i in range(0, len(x_data), batch_size):
        start, stop = i, i + batch_size
        x_batch = batch_to_ids(x_data[start:stop])
        lengths = Variable(torch.from_numpy(np.array([max(len(x), 1) for x in x_data[start:stop]])).float()).view(-1, 1)
        if CUDA:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])).cuda())
        else:
            y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])))
        batches.append((x_batch, y_batch, lengths))

    if shuffle:
        random.shuffle(batches)

    return batches


def host_is_up(hostname, port, protocol='http'):
    url = f'{protocol}://{hostname}:{port}'
    try:
        request.urlopen(url).getcode()
        return True
    except request.URLError:
        return False


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
        mask = Variable(mask)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight

    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    X = embed._backend.Embedding.apply(
        words, masked_embed_weight,
        padding_idx, embed.max_norm, embed.norm_type,
        embed.scale_grad_by_freq, embed.sparse
    )
    return X


def create_save_model(model):
    def save_model(path):
        torch.save(model, path)
    return save_model


class Callback(abc.ABC):
    @abc.abstractmethod
    def on_epoch_end(self, logs) -> Tuple[bool, Optional[str]]:
        pass


class BaseLogger(Callback):
    def __init__(self, log_func=print):
        self.log_func = log_func
    def on_epoch_end(self, logs):
        msg = 'Epoch {}: train_acc={:.4f} test_acc={:.4f} | train_loss={:.4f} test_loss={:.4f} | time={:.1f}'.format(
            len(logs['train_acc']),
            logs['train_acc'][-1], logs['test_acc'][-1],
            logs['train_loss'][-1], logs['test_loss'][-1],
            logs['train_time'][-1]
        )
        self.log_func(msg)

    def __repr__(self):
        return 'BaseLogger()'


class TerminateOnNaN(Callback):
    def on_epoch_end(self, logs):
        for _, arr in logs.items():
            if np.any(np.isnan(arr)):
                raise ValueError('NaN encountered')
        else:
            return False, None

    def __repr__(self):
        return 'TerminateOnNaN()'


class EarlyStopping(Callback):
    def __init__(self, monitor='test_loss', min_delta=0, patience=1, verbose=0, log_func=print):
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.current_patience = patience
        self.verbose = verbose
        self.log_func = log_func

    def __repr__(self):
        return 'EarlyStopping(monitor={}, min_delta={}, patience={})'.format(
            self.monitor, self.min_delta, self.patience)

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.current_patience = self.patience
            self.best_monitor_score = logs[self.monitor][-1]
        else:
            self.current_patience -= 1
            if self.verbose > 0:
                self.log_func('Patience: reduced by one and waiting for {} epochs for improvement before stopping'.format(self.current_patience))

        if self.current_patience == 0:
            return True, 'Ran out of patience'
        else:
            return False, None


class MaxEpochStopping(Callback):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs

    def on_epoch_end(self, logs):
        if len(logs['train_time']) == self.max_epochs:
            return True, 'Max epochs reached'
        else:
            return False, None


class ModelCheckpoint(Callback):
    def __init__(self, save_function, filepath, monitor='test_loss', save_best_only=True, verbose=0, log_func=print):
        self.save_function = save_function
        self.filepath = filepath
        self.save_best_only = save_best_only
        if monitor.endswith('loss'):
            self.improvement_sign = 1
        elif monitor.endswith('acc'):
            self.improvement_sign = -1
        else:
            raise ValueError('Unrecognized monitor')
        self.monitor = monitor
        self.best_monitor_score = self.improvement_sign * float('inf')
        self.verbose = verbose
        self.log_func = log_func

    def on_epoch_end(self, logs):
        if logs[self.monitor][-1] * self.improvement_sign < self.improvement_sign * self.best_monitor_score:
            self.best_monitor_score = logs[self.monitor][-1]
            if self.save_best_only:
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(self.filepath))
                self.save_function(self.filepath)
            else:
                path = self.filepath.format(epoch=len(logs['train_time']) - 1)
                if self.verbose > 0:
                    self.log_func('New best model, saving to: {}'.format(path))
                self.save_function(path)


class TrainingManager:
    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks
        self.logs = defaultdict(list)

    def instruct(self, train_time, train_loss, train_acc, test_time, test_loss, test_acc):
        self.logs['train_time'].append(train_time)
        self.logs['train_loss'].append(train_loss)
        self.logs['train_acc'].append(train_acc)
        self.logs['test_time'].append(test_time)
        self.logs['test_loss'].append(test_loss)
        self.logs['test_acc'].append(test_acc)

        callback_stop_reasons = []
        for c in self.callbacks:
            result = c.on_epoch_end(self.logs)
            if result is None:
                stop_training, reason = False, None
            else:
                stop_training, reason = result
            if stop_training:
                callback_stop_reasons.append('{}: {}'.format(c.__class__.__name__, reason))

        if len(callback_stop_reasons) > 0:
            return True, callback_stop_reasons
        else:
            return False, []


from tqdm import tqdm
def run_epoch(batches, train=True):
    batch_accuracies = []
    batch_losses = []
    epoch_start = time.time()
    for i in tqdm(range(len(batches))):
        x_batch, y_batch, length_batch = batches[i]
        if train:
            model.zero_grad()
        out = model(x_batch.cuda(), length_batch.cuda())
#         out = model(x_batch, length_batch)
        _, preds = torch.max(out, 1)
        accuracy = torch.mean(torch.eq(preds, y_batch).float()).data[0]
        batch_loss = criterion(out, y_batch)
        if train:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), .25)
            optimizer.step()
        batch_accuracies.append(accuracy)
        batch_losses.append(batch_loss.data[0])
    epoch_end = time.time()

    return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start


class ElmoGuesser():
    def __init__(self):
        super(ElmoGuesser, self).__init__()
        self.random_seed = 1
        self.dropout = 0.5

        self.model = None
        self.i_to_class = None
        self.class_to_i = None

        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.model_file = None


    def train(self, training_data: TrainingData) -> None:
        x_train, y_train, x_val, y_val, vocab, class_to_i, i_to_class = preprocess_dataset(training_data)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class

        print('Batchifying data')
        train_batches = batchify(x_train, y_train, shuffle=True)
        val_batches = batchify(x_val, y_val, shuffle=False)
        self.model = ElmoModel(len(i_to_class), dropout=self.dropout)
        if CUDA:
            self.model = self.model.cuda()
        print(f'Model:\n{self.model}')
        parameters = list(self.model.classifier.parameters())
        for mix in self.model.elmo._scalar_mixes:
            parameters.extend(list(mix.parameters()))
        self.optimizer = Adam(parameters)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')
        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'
        manager = TrainingManager([
            BaseLogger(log_func=logging.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])
        logging.info('Starting training')
        epoch = 0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_batches)
            random.shuffle(train_batches)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(val_batches)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                logging.info(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)
            epoch += 1

    def run_epoch(self, batches, train=True):
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for x_batch, y_batch, length_batch in batches:
            if train:
                self.model.zero_grad()
            out = self.model(x_batch.cuda(), length_batch.cuda())
            #out = self.model(x_batch, length_batch)
            _, preds = torch.max(out, 1)
            accuracy = torch.mean(torch.eq(preds, y_batch).float()).data[0]
            batch_loss = self.criterion(out, y_batch)
            if train:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), .25)
                self.optimizer.step()
            batch_accuracies.append(accuracy)
            batch_losses.append(batch_loss.data[0])
        epoch_end = time.time()

        return np.mean(batch_accuracies), np.mean(batch_losses), epoch_end - epoch_start

    def guess(self, questions: List[QuestionText], max_n_guesses: Optional[int]) -> List[List[Tuple[Page, float]]]:
        y_data = np.zeros((len(questions)))
        x_data = [tokenize_question(q) for q in questions]
        batches = batchify(x_data, y_data, shuffle=False, batch_size=32)
        guesses = []
        for x_batch, y_batch, length_batch in batches:
            out = self.model(x_batch.cuda(), length_batch.cuda())
            #out = self.model(x_batch, length_batch)
            probs = F.softmax(out).data.cpu().numpy()
            preds = np.argsort(-probs, axis=1)
            n_examples = probs.shape[0]
            for i in range(n_examples):
                example_guesses = []
                for p in preds[i][:max_n_guesses]:
                    example_guesses.append((self.i_to_class[p], probs[i][p]))
                guesses.append(example_guesses)

        return guesses

    @classmethod
    def targets(cls) -> List[str]:
        return ['elmo.pt', 'elmo.pkl']

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'elmo.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = ElmoGuesser(params['config_num'])
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.random_seed = params['random_seed']
        guesser.dropout = params['dropout']
        guesser.model = ElmoModel(len(guesser.i_to_class))
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'elmo.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        if CUDA:
            guesser.model = guesser.model.cuda()
        return guesser

    def save(self, directory: str) -> None:
        shutil.copyfile(self.model_file, os.path.join(directory, 'elmo.pt'))
        shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'elmo.pkl'), 'wb') as f:
            cloudpickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'config_num': self.config_num,
                'random_seed': self.random_seed,
                'dropout': self.dropout
            }, f)

def main():
    dataset = QuizBowlDataset(guesser_train=True)
    training_data = dataset.training_data()

    elm = ElmoGuesser()
    elm.train(training_data)

if __name__ == '__main__':
    main()
