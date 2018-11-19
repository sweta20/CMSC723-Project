
from preprocess import preprocess_dataset, WikipediaDataset
from dataset import QuizBowlDataset
from util import create_save_model
from models import ElmoModel
from util import BaseLogger, TerminateOnNaN, EarlyStopping, ModelCheckpoint, MaxEpochStopping, TrainingManager
from util import get, get_tmp_filename
from util import QuestionText, TrainingData, Page, Evidence
import argparse
import json
import numpy as np
from typing import List, Optional, Tuple
import os
import shutil
import random
import time
import cloudpickle
import torch
import torch.nn as nn
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler

categories = {
    0: ['History', 'Philosophy', 'Religion'],
    1: ['Literature', 'Mythology'],
    2: ['Science', 'Social Science'],
    3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
}

parser = argparse.ArgumentParser(description='SkinTone Embedding')
parser.add_argument('--full_question', type=bool, default=False,
                    help='Use full question (default: False)')
parser.add_argument('--create_runs', type=bool, default=False,
                    help='Use full question (default: False)')
parser.add_argument('--category', type=int, default=None,
                    help='''categories = {
                            0: ['History', 'Philosophy', 'Religion'],
                            1: ['Literature', 'Mythology'],
                            2: ['Science', 'Social Science'],
                            3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
                            } (default:None)''')
parser.add_argument('--use_wiki', type=bool, default=False,
                    help='use_wiki (default: False)')
parser.add_argument('--n_wiki_sentences', type=int, default=5,
                    help='n_wiki_sentences (default: 5)')


def get_quizbowl(guesser_train=True, buzzer_train=False, category=None, use_wiki=False, n_wiki_sentences = 5):
    print("Loading data with guesser_train: " + str(guesser_train) + " buzzer_train:  " + str(buzzer_train))
    qb_dataset = QuizBowlDataset(guesser_train=guesser_train, buzzer_train=buzzer_train, category=category)
    training_data = qb_dataset.training_data()
    
    if use_wiki and n_wiki_sentences > 0:
        print("Using wiki dataset with n_wiki_sentences: " + str(n_wiki_sentences))
        wiki_dataset = WikipediaDataset(set(training_data[1]), n_wiki_sentences)
        wiki_training_data = wiki_dataset.training_data()
        training_data[0].extend(wiki_training_data[0])
        training_data[1].extend(wiki_training_data[1])
    return training_data

def batchify(x_data, y_data, batch_size=32, shuffle=False):
    batches = []
    for i in range(0, len(x_data), batch_size):
        start, stop = i, i + batch_size
        x_batch = batch_to_ids(x_data[start:stop])
        lengths = Variable(torch.from_numpy(np.array([max(len(x), 1) for x in x_data[start:stop]])).float()).view(-1, 1)
        y_batch = Variable(torch.from_numpy(np.array(y_data[start:stop])))
        batches.append((x_batch, y_batch, lengths))

    if shuffle:
        random.shuffle(batches)

    return batches

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def train(self, training_data: TrainingData) -> None:
        x_train, y_train, x_val, y_val, vocab, class_to_i, i_to_class = preprocess_dataset(training_data, full_question=args.full_question, create_runs=args.create_runs)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class

        print('Batchifying data')
        train_batches = batchify(x_train, y_train, shuffle=True)
        val_batches = batchify(x_val, y_val, shuffle=False)
        self.model = ElmoModel(len(i_to_class), dropout=self.dropout)
        self.model = self.model.to(self.device)
        
        print(f'Model:\n{self.model}')
        parameters = list(self.model.classifier.parameters())
        for mix in self.model.elmo._scalar_mixes:
            parameters.extend(list(mix.parameters()))
        self.optimizer = Adam(parameters)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')


        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'

        print(f'Saving model to: {self.model_file}')
        log = get(__name__)
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(100), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        log.info('Starting training')

        epoch = 0
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_batches)
            random.shuffle(train_batches)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(val_batches, train=False)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if stop_training:
                log.info(' '.join(reasons))
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
            y_batch = y_batch.to(self.device)
            out = self.model(x_batch.to(self.device), length_batch.to(self.device))
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
            y_batch = y_batch.to(self.device)
            out = self.model(x_batch.to(self.device), length_batch.to(self.device))
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

        guesser = ElmoGuesser()
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.random_seed = params['random_seed']
        guesser.dropout = params['dropout']
        guesser.model = ElmoModel(len(guesser.i_to_class))
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'elmo.pt'), map_location=lambda storage, loc: storage
        ))
        guesser.model.eval()
        guesser.model = guesser.model.to(self.device)
        return guesser

    def save(self, directory: str) -> None:
        shutil.copyfile(self.model_file, os.path.join(directory, 'elmo.pt'))
       # shell(f'rm -f {self.model_file}')
        with open(os.path.join(directory, 'elmo.pkl'), 'wb') as f:
            cloudpickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'random_seed': self.random_seed,
                'dropout': self.dropout
            }, f)

def main():

    global args
    args = parser.parse_args()
    category = categories[args.category] if args.category is not None else None

    training_data = get_quizbowl(category=category, use_wiki=args.use_wiki, n_wiki_sentences = args.n_wiki_sentences)

    elm = ElmoGuesser()
    elm.train(training_data)

    elm.save("./")


if __name__ == '__main__':
    main()
