import json


import numpy as np
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('resources/intents.json', 'r') as f:
    intents = json.load(f)

# print(intents)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        # extend because w is an array
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ["?", "!", ".", ","]
all_words = [stem(w) for w in all_words if w not in ignore_words]

all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)  # one hot encoding generally used ,but later will use cross entropy loss

X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatBotDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# hyper parameters


batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


# print(input_size,len(all_words))
# print(output_size,tags)

dataset = ChatBotDataset()
#Num workers reduced to zero from two due to cuda multiprocessing error
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet(input_size, hidden_size, output_size).to(device)


# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for [words, labels] in train_loader:
        words, labels = words.to(device), labels.to(device)

        # forward pass
        outputs = model(words)

        loss = criterion(outputs, labels)

        # backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if ((epoch + 1) % 100) == 0:
        print(f'Epoch {epoch + 1} / {num_epochs}, Loss {loss.item():.4f}')
print(f'final loss: {loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

# pth = pytorch
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')

