from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import random


def train_model_2():
    import pickle
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1  # Plus EOS marker

    def findFiles(path): return glob.glob(path)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    # Read a file and split into lines
    def readLines(filename):
        with open(filename, encoding='utf-8') as some_file:
            return [unicodeToAscii(line.strip()) for line in some_file]

    # Build the category_lines dictionary, a list of lines per category
    category_lines = {}
    all_categories = []
    for filename in findFiles('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)

    if n_categories == 0:
        raise RuntimeError('Data not found. Make sure that you downloaded data '
            'from https://download.pytorch.org/tutorial/data.zip and extract it to '
            'the current directory.')

    print('# categories:', n_categories, all_categories)
    print(unicodeToAscii("O'Néàl"))



    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size

            self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
            self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
            self.o2o = nn.Linear(hidden_size + output_size, output_size)
            self.dropout = nn.Dropout(0.1)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, category, input, hidden):
            input_combined = torch.cat((category, input, hidden), 1)
            hidden = self.i2h(input_combined)
            output = self.i2o(input_combined)
            output_combined = torch.cat((hidden, output), 1)
            output = self.o2o(output_combined)
            output = self.dropout(output)
            output = self.softmax(output)
            return output, hidden

        def initHidden(self):
            return torch.zeros(1, self.hidden_size)



    # Random item from a list
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    # Get a random category and random line from that category
    def randomTrainingPair():
        category = randomChoice(all_categories)
        line = randomChoice(category_lines[category])
        return category, line

    # One-hot vector for category
    def categoryTensor(category):
        li = all_categories.index(category)
        tensor = torch.zeros(1, n_categories)
        tensor[0][li] = 1
        return tensor

    # One-hot matrix of first to last letters (not including EOS) for input
    def inputTensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor

    # LongTensor of second letter to end (EOS) for target
    def targetTensor(line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(n_letters - 1) # EOS
        return torch.LongTensor(letter_indexes)

    # Make category, input, and target tensors from a random category, line pair
    def randomTrainingExample():
        category, line = randomTrainingPair()
        category_tensor = categoryTensor(category)
        input_line_tensor = inputTensor(line)
        target_line_tensor = targetTensor(line)
        return category_tensor, input_line_tensor, target_line_tensor


    # criterion = nn.NLLLoss()
    #
    # learning_rate = 0.0005

    def train(n_iters,plot_every,rnn):
        # rnn = RNN(n_letters, 128, n_letters)
        all_losses = []
        total_loss = 0
        loss = 0

        for iter in range(1, n_iters + 1):
            category_tensor,input_line_tensor,target_line_tensor = randomTrainingExample()
            target_line_tensor.unsqueeze_(-1)

            hidden = rnn.initHidden()
            rnn.zero_grad()

            loss = 0
            for i in range(input_line_tensor.size(0)):
                output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
                l = criterion(output, target_line_tensor[i])
                loss += l

            loss.backward()
            for p in rnn.parameters():
                p.data.add_(p.grad.data, alpha=-learning_rate)


            total_loss += loss.item() / input_line_tensor.size(0)

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0

        plt.figure()
        plt.title("RNN Loss evry 1000 iters")
        plt.ylabel("Loss")
        plt.xlabel("Epochs/1000")
        plt.plot(all_losses)
        plt.show()

        # with open('model_q2.pkl', 'wb') as f:
        #     pickle.dump(rnn, f)





    # def train_model_q2():
    #     # all_letters = string.ascii_letters + " .,;'-"
    #     # n_letters = len(all_letters) + 1  # Plus EOS marker
    #     # category_lines = {}
    #     # all_categories = []
    #     for filename in findFiles('data/names/*.txt'):
    #         category = os.path.splitext(os.path.basename(filename))[0]
    #         all_categories.append(category)
    #         lines = readLines(filename)
    #         category_lines[category] = lines
    #     n_categories = len(all_categories)

    criterion = nn.NLLLoss()
    learning_rate = 0.0005

    rnn = RNN(n_letters, 128, n_letters)
    n_iters = 100000
    plot_every = 500  # Reset every plot_every iters
    train(n_iters, plot_every,rnn)
    # torch.save(rnn.state_dict(), 'model_q2.pkl')

# train_model_q2()