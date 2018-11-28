from tqdm import tqdm
from preprocess import preprocess_dataset, WikipediaDataset, tokenize_question
from dataset import QuizBowlDataset
from util import create_save_model
from models import DanModel
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
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F


categories = {
    0: ['History', 'Philosophy', 'Religion'],
    1: ['Literature', 'Mythology'],
    2: ['Science', 'Social Science'],
    3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='DAN training')
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
parser.add_argument('--batch_size', type=int, default=32,
                    help='batch_size (default: 32)')
parser.add_argument('--save_model', type=str, default="dan.pt",
                    help='save_model (default: dan.pt)')
parser.add_argument('--eval', default=False, action='store_true',
                    help='Run the evalulation')


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return np.array(ids, 'i')


def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), np.array([cls], 'i'))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]


def get_quizbowl(guesser_train=True, buzzer_train=False, category=None, use_wiki=False, n_wiki_sentences=5):
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


def load_glove(filename):
    idx = 0
    word2idx = {}
    vectors = []

    with open(filename, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)

    return word2idx, vectors


class DANGuesser():
    def __init__(self):
        super(DANGuesser, self).__init__()
        self.model = None
        self.i_to_class = None
        self.class_to_i = None

        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.model_file = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batchify(self, batch):
        """
        Gather a batch of individual examples into one batch, 
        which includes the question text, question length and labels 

        Keyword arguments:
        batch: list of outputs from vectorize function
        """
        batch = transform_to_array(batch, self.word_to_i)
        question_len = list()
        label_list = list()
        for ex in batch:
            question_len.append(len(ex[0]))
            label_list.append(ex[1][0])
        target_labels = torch.LongTensor(label_list)
        x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
        for i in range(len(question_len)):
            question_text = batch[i][0]
            vec = torch.LongTensor(question_text)
            x1[i, :len(question_text)].copy_(vec)
        q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
        return q_batch

    def train(self, training_data: TrainingData) -> None:
        x_train, y_train, x_val, y_val, i_to_word, class_to_i, i_to_class = preprocess_dataset(training_data,
                                                                                               full_question=args.full_question,
                                                                                               create_runs=args.create_runs)
        self.class_to_i = class_to_i
        self.i_to_class = i_to_class
        # log = get(__name__, "dan.log")
        # log.info('Batchifying data')
        print('Batchifying data')
        i_to_word = ['<unk>', '<eos>'] + sorted(i_to_word)
        word_to_i = {x: i for i, x in enumerate(i_to_word)}
        self.word_to_i = word_to_i
        # log.info('Vocab len: ' + str(len(self.word_to_i)))
        print('Vocab len: ' + str(len(self.word_to_i)))

        train_sampler = RandomSampler(list(zip(x_train, y_train)))
        dev_sampler = RandomSampler(list(zip(x_val, y_val)))
        dev_loader = DataLoader(list(zip(x_val, y_val)), batch_size=args.batch_size,
                                sampler=dev_sampler, num_workers=0,
                                collate_fn=self.batchify)
        train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=args.batch_size,
                                  sampler=train_sampler, num_workers=0,
                                  collate_fn=self.batchify)

        self.model = DanModel(len(i_to_class), len(i_to_word))
        self.model = self.model.to(self.device)

        # log.info(f'Loading GloVe')
        print('Loading GloVe')
        glove_word2idx, glove_vectors = load_glove("glove/glove.6B.300d.txt")
        for word, emb_index in word_to_i.items():
            if word.lower() in glove_word2idx:
                glove_index = glove_word2idx[word.lower()]
                glove_vec = torch.FloatTensor(glove_vectors[glove_index])
                glove_vec = glove_vec.cuda()
                self.model.text_embeddings.weight.data[emb_index, :].set_(glove_vec)

        # log.info(f'Model:\n{self.model}')
        print('Model:\n{self.model}')
        self.optimizer = Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, verbose=True, mode='max')

        temp_prefix = get_tmp_filename()
        self.model_file = f'{temp_prefix}.pt'

        print(f'Saving model to: {self.model_file}')
        log = get(__name__)
        manager = TrainingManager([
            BaseLogger(log_func=log.info), TerminateOnNaN(), EarlyStopping(monitor='test_acc', patience=10, verbose=1),
            MaxEpochStopping(500), ModelCheckpoint(create_save_model(self.model), self.model_file, monitor='test_acc')
        ])

        print('Starting training')

        epoch = 0
        accuracy = 0.0
        state = {}
        while True:
            self.model.train()
            train_acc, train_loss, train_time = self.run_epoch(train_loader)

            self.model.eval()
            test_acc, test_loss, test_time = self.run_epoch(dev_loader, train=False)

            stop_training, reasons = manager.instruct(
                train_time, train_loss, train_acc,
                test_time, test_loss, test_acc
            )

            if accuracy < test_acc:
                accuracy = test_acc
                print('Saving..')
                state = {
                    'net': self.model.state_dict(),
                    'acc': accuracy,
                    'epoch': epoch,
                }

            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.t7')

            if stop_training:
                # log.info(' '.join(reasons))
                print(' '.join(reasons))
                break
            else:
                self.scheduler.step(test_acc)
            epoch += 1

    def run_epoch(self, data_loader, train=True):
        batch_accuracies = []
        batch_losses = []
        epoch_start = time.time()
        for idx, batch in tqdm(enumerate(data_loader)):
            x_batch = batch['text'].to(self.device)
            length_batch = batch['len'].to(self.device)
            y_batch = batch['labels'].to(self.device)
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
        batches = self.batchify(x_data, y_data, shuffle=False, batch_size=32)
        guesses = []
        for x_batch, y_batch, length_batch in batches:
            y_batch = y_batch.to(self.device)
            out = self.model(x_batch.to(self.device), length_batch.to(self.device))
            probs = F.softmax(out).cpu().numpy()
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
        return ['dan.pt', 'dan.pkl']

    @classmethod
    def load(cls, directory: str):
        with open(os.path.join(directory, 'dan.pkl'), 'rb') as f:
            params = cloudpickle.load(f)

        guesser = DANGuesser()
        guesser.class_to_i = params['class_to_i']
        guesser.i_to_class = params['i_to_class']
        guesser.word_to_i = params['word_to_i']
        guesser.device = params['device']
        guesser.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        guesser.model = DanModel(len(guesser.i_to_class), len(guesser.word_to_i))
        guesser.model.load_state_dict(torch.load(
            os.path.join(directory, 'dan.pt'), map_location=lambda storage, loc: storage
        ).state_dict())
        guesser.model.eval()
        guesser.model = guesser.model.to(guesser.device)
        return guesser

    def save(self, directory: str) -> None:
        shutil.copyfile(self.model_file, os.path.join(directory, 'dan.pt'))
        with open(os.path.join(directory, 'dan.pkl'), 'wb') as f:
            cloudpickle.dump({
                'class_to_i': self.class_to_i,
                'i_to_class': self.i_to_class,
                'word_to_i': self.word_to_i,
                'device': self.device
            }, f)


def main():
    global args
    args = parser.parse_args()
    category = categories[args.category] if args.category is not None else None
    if args.eval:
        dataset = QuizBowlDataset(guesser_train=True)
        questions = dataset.questions_by_fold()
        questions = questions['guessdev']
        dan = DANGuesser().load("./")
        dan.guess(questions)

    else:
        training_data = get_quizbowl(category=category, use_wiki=args.use_wiki, n_wiki_sentences=args.n_wiki_sentences)

        dan = DANGuesser()
        dan.train(training_data)

        dan.save("./")


if __name__ == '__main__':
    main()
