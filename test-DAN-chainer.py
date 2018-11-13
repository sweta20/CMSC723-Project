#!/usr/bin/env python
# coding: utf-8

from preprocess import preprocess_dataset
from dataset import QuizBowlDataset
import json
import collections
import io
import numpy
import chainer
from chainer import cuda
import os
import numpy as np
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter
import chainer
from chainer import training
from chainer.training import extensions
import datetime

embed_init = chainer.initializers.Uniform(.25)

def get_quizbowl():
    qb_dataset = QuizBowlDataset(guesser_train=True, buzzer_train=False)
    training_data = qb_dataset.training_data()
    train_x, train_y, dev_x, dev_y, i_to_word, class_to_i, i_to_class = preprocess_dataset(training_data)
    i_to_word = ['<unk>', '<eos>'] + sorted(i_to_word)
    word_to_i = {x: i for i, x in enumerate(i_to_word)}
    train = transform_to_array(zip(train_x, train_y), word_to_i)
    dev = transform_to_array(zip(dev_x, dev_y), word_to_i)
    return train, dev, word_to_i, i_to_class


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab['<unk>']
    eos_id = vocab['<eos>']
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, 'i')


def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), numpy.array([cls], 'i'))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]


def convert_seq(batch, device=None, with_label=True):
    def to_device_batch(batch):
        if device is None:
            return batch
        elif device < 0:
            return [chainer.dataset.to_device(device, x) for x in batch]
        else:
            xp = cuda.cupy.get_array_module(*batch)
            concat = xp.concatenate(batch, axis=0)
            sections = numpy.cumsum([len(x) for x in batch[:-1]], dtype='i')
            concat_dev = chainer.dataset.to_device(device, concat)
            batch_dev = cuda.cupy.split(concat_dev, sections)
            return batch_dev

    if with_label:
        return {'xs': to_device_batch([x for x, _ in batch]),
                'ys': to_device_batch([y for _, y in batch])}
    else:
        return to_device_batch([x for x in batch])



def sequence_embed(embed, xs, dropout=0.):
    """Efficient embedding function for variable-length sequences

    This output is equally to
    "return [F.dropout(embed(x), ratio=dropout) for x in xs]".
    However, calling the functions is one-shot and faster.

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        xs (list of :class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): i-th element in the list is an input variable,
            which is a :math:`(L_i, )`-shaped int array.
        dropout (float): Dropout ratio.

    Returns:
        list of ~chainer.Variable: Output variables. i-th element in the
        list is an output variable, which is a :math:`(L_i, N)`-shaped
        float array. :math:`(N)` is the number of dimensions of word embedding.

    """
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    ex = F.dropout(ex, ratio=dropout)
    exs = F.split_axis(ex, x_section, 0)
    return exs


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class DANEncoder(chainer.Chain):

    def __init__(self, n_vocab, embed_size, hidden_size, dropout):
        super(DANEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, embed_size, ignore_label=-1,
                    initialW=embed_init)
            self.linear = L.Linear(embed_size, hidden_size)
            self.batchnorm = L.BatchNormalization(hidden_size)
        self.dropout = dropout
        self.output_size = hidden_size
    
    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], 'i')[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len

        h = self.linear(h)
        h = self.batchnorm(h)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        return h


class NNGuesser(chainer.Chain):

    def __init__(self, encoder, n_class, dropout):
        super(NNGuesser, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.linear = L.Linear(encoder.output_size, n_class)
            self.batchnorm = L.BatchNormalization(n_class)
            # dropout
        self.dropout = dropout

    def load_glove(self, raw_path, vocab, size):
        print('Constructing embedding matrix')
        embed_w = np.random.uniform(-0.25, 0.25, size)
        with open(raw_path, 'r') as f:
            for line in tqdm(f):
                line = line.strip().split(" ")
                word = line[0]
                if word in vocab:
                    vec = np.array(line[1::], dtype=np.float32)
                    embed_w[vocab[word]] = vec
        embed_w = self.xp.array(embed_w, dtype=self.xp.float32)
        self.encoder.embed.W.data = embed_w
    
    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        h = self.encoder(xs)

        h = self.linear(h)
        h = self.batchnorm(h)
        h = F.dropout(h, ratio=self.dropout)

        if softmax:
            return F.softmax(h).data
        elif argmax:
            return self.xp.argmax(h.data, axis=1)
        else:
            return h

def main():

    train, dev, vocab, answers = get_quizbowl()
    n_vocab = len(vocab)
    n_class = len(set([int(d[1]) for d in train]))
    embed_size = 300
    hidden_size = 512
    hidden_dropout = 0.3
    output_dropout = 0.2
    gradient_clipping = 0.25


    print('# train data: {}'.format(len(train)))
    print('# dev data: {}'.format(len(dev)))
    print('# vocab: {}'.format(len(vocab)))
    print('# class: {}'.format(n_class))
    print('embedding size: {}'.format(embed_size))
    print('hidden size: {}'.format(hidden_size))
    print('hidden dropout: {}'.format(hidden_dropout))
    print('output dropout: {}'.format(output_dropout))
    print('gradient clipping: {}'.format(gradient_clipping))


    batchsize= 64
    epoch = 30
    gpu=0
    out = "result"
    model = "dan"
    glove = "/home/sweta/Work/CMSC723/project/glove/glove.6B.300d.txt"


    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    dev_iter = chainer.iterators.SerialIterator(dev, batchsize,  repeat=False, shuffle=False)


    encoder = DANEncoder(n_vocab, embed_size, hidden_size,
                    dropout=hidden_dropout)
    model = NNGuesser(encoder, n_class, dropout=output_dropout)
    model.load_glove(glove, vocab, (n_vocab, embed_size))
    if gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()

    chainer.backends.cuda.available

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(gradient_clipping))

    updater = training.StandardUpdater(
    train_iter, optimizer,
    converter=convert_seq, device=gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

    trainer.extend(extensions.Evaluator(
        dev_iter, model,
        converter=convert_seq, device=gpu))

    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy',
             'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if not os.path.isdir(out):
        os.mkdir(out)
    current_datetime = '{}'.format(datetime.datetime.today())    
    # current = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(out, 'vocab.json')
    answers_path = os.path.join(out, 'answers.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    with open(answers_path, 'w') as f:
        json.dump(answers, f)
    model_path = os.path.join(out, 'best_model.npz')
    model_setup = {}
    model_setup['vocab_path'] = vocab_path
    model_setup['answers_path'] = answers_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(out, 'args.json'), 'w') as f:
        json.dump(model_setup, f)

    # if resume:
    #     print('loading model {}'.format(model_path))
    #     chainer.serializers.load_npz(model_path, model)

    # Run the training
    trainer.run()



if __name__ == '__main__':
    main()
