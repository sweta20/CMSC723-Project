#!/usr/bin/env python
# coding: utf-8

# In[1]:


from preprocess import preprocess_dataset, tokenize_question
from dataset import QuizBowlDataset
import collections
import io
import random
import numpy
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_


# In[8]:



categories = {
    0: ['History', 'Philosophy', 'Religion'],
    1: ['Literature', 'Mythology'],
    2: ['Science', 'Social Science'],
    3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
}

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

def get_quizbowl():
    qb_dataset = QuizBowlDataset(guesser_train=True, buzzer_train=False)
    training_data = qb_dataset.training_data()
    train_x, train_y, dev_x, dev_y, i_to_word, class_to_i, i_to_class = preprocess_dataset(training_data)
    i_to_word = ['<unk>', '<eos>'] + sorted(i_to_word)
    word_to_i = {x: i for i, x in enumerate(i_to_word)}
    train = transform_to_array(zip(train_x, train_y), word_to_i)
    dev = transform_to_array(zip(dev_x, dev_y), word_to_i)
    return train, dev, word_to_i, i_to_class


# In[9]:


def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set

    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    decide: cpu of gpu
    """

    model.eval()
    with torch.no_grad():
        num_examples = 0
        error = 0
        for idx, batch in enumerate(data_loader):
            question_text = batch['text'].to(device)
            question_len = batch['len'].to(device)
            labels = batch['labels'].to(device)
            ####Your code here
            logits = model(question_text, question_len)

            top_n, top_i = logits.topk(1)
#             print([answers[t] for t in top_i.cpu().data[0].numpy()], [answers[t] for t in labels.cpu().data[0].numpy() ])
            num_examples += question_text.size(0)

            error += torch.nonzero(top_i.squeeze() -  torch.max(labels, 1)[0]).size(0)

        accuracy = 1 - error / num_examples
        print('accuracy', accuracy)
    return accuracy


# In[10]:


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


# In[11]:


class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model


    def __init__(self, n_classes, vocab_size, emb_dim=300,
                 n_hidden_units=300, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        #### modify the init function, you need to add necessary layer definition here
        #### note that linear1, linear2 are used for mlp layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(1)
       


    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        in_prob: if True, output the softmax of last layer

        """
        #### write the forward funtion, the output is logits 
        embeds = self.embeddings(input_text)
        encoded =  embeds.sum(1)
        encoded /= text_len.view(embeds.size(0), -1)
        logits = self.linear2(self.relu(self.linear1(encoded)))
        if is_prob:
            return  self.softmax(logits)
        else:
            return logits


# In[13]:


grad_clipping = 5
save_model = "dan_1.pt"
checkpoint = 100
 
def train_model(model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model

    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    decide: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len'].to(device)
        labels = batch['labels'].to(device)

        #### Your code here
        probs = model(question_text, question_len)
        loss = criterion(probs, torch.max(labels, 1)[0])
        loss.backward()
        optimizer.step()

        clip_grad_norm_(model.parameters(), grad_clipping)
        print_loss_total += loss.cpu().data.numpy()
        epoch_loss_total += loss.cpu().data.numpy()

        if idx % checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, save_model)
                accuracy = curr_accuracy
    return accuracy


# In[14]:


def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 

    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


# In[15]:


train, dev, vocab, answers = get_quizbowl()
random.shuffle(train)
random.shuffle(dev)
n_vocab = len(vocab)
n_class = len(set([int(d[1]) for d in train]))
print('# train data: {}'.format(len(train)))
print('# dev data: {}'.format(len(dev)))


# In[16]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[17]:


model = DanModel(len(answers), len(vocab))
model.to(device)


# In[18]:


glove_word2idx, glove_vectors = load_glove("glove/glove.6B.300d.txt")


# In[19]:


for word, emb_index in vocab.items():
    if word.lower() in glove_word2idx:
        glove_index = glove_word2idx[word.lower()]
        glove_vec = torch.FloatTensor(glove_vectors[glove_index])
        glove_vec = glove_vec.cuda()

        model.embeddings.weight.data[emb_index, :].set_(glove_vec)


# In[20]:


batch_size = 32
train_sampler = torch.utils.data.sampler.RandomSampler(train)
dev_sampler = torch.utils.data.sampler.SequentialSampler(dev)
dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                               sampler=dev_sampler, num_workers=0,
                                               collate_fn=batchify)


# In[21]:


num_epochs = 1000
accuracy = 0
for epoch in range(num_epochs):
    print('start epoch %d' % epoch)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                       sampler=train_sampler, num_workers=0,
                                       collate_fn=batchify)
    accuracy = train_model(model, train_loader, dev_loader, accuracy, device)


# In[ ]:


model = torch.load(save_model)


# In[ ]:



model = torch.load(save_model)
#### Load batchifed dataset
print('start testing on dev set:\n')

dev_sampler = torch.utils.data.sampler.SequentialSampler(dev)
dev_loader = torch.utils.data.DataLoader(dev, batch_size=batch_size,
                                       sampler=dev_sampler, num_workers=0,
                                       collate_fn=batchify)
evaluate(dev_loader, model, device)


# In[ ]:


batch = next(iter(dev_loader))


# In[ ]:


model.eval()
question_text = batch['text'].to(device)
question_len = batch['len'].to(device)
labels = batch['labels'].to(device)
####Your code here
logits = model(question_text, question_len)
top_n, top_i = logits.topk(1)


# In[ ]:


q = "Name the inventor of general relativity and the photoelectric effect"


# In[ ]:


x_data = [tokenize_question(q)]
x_data = transform_to_array(x_data, vocab, with_label=False)

question_len = list()
for ex in x_data:
    question_len.append(len(ex))
x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
question_text = x_data[0]
vec = torch.LongTensor(question_text)
x1[0, :len(question_text)].copy_(vec)
q_batch = {'text': x1, 'len': torch.FloatTensor(question_len)}

prob = model(q_batch['text'].to(device), q_batch['len'].to(device))
top_n, top_i = prob.topk(1)
print(answers[top_i.cpu().data[0].numpy()[0]])


# In[ ]:





# In[ ]:




