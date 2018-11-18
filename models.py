import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from allennlp.modules.elmo import Elmo, batch_to_ids
import torch

ELMO_OPTIONS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
ELMO_WEIGHTS_FILE = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
ELMO_DIM = 1024

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

class ElmoDANModel(nn.Module):
    def __init__(self, n_classes, n_hidden_units, dropout=.25):
        super().__init__()
        self.dropout = dropout
        # This turns off gradient updates for the elmo model, but still leaves scalar mixture
        # parameters as tunable, provided that references to the scalar mixtures are extracted
        # and plugged into the optimizer
        self.elmo = Elmo(ELMO_OPTIONS_FILE, ELMO_WEIGHTS_FILE, 2, dropout=dropout, requires_grad=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * ELMO_DIM, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_classes),
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
