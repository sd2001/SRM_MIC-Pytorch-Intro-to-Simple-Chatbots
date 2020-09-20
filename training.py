import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from basics import bag_of_words, tokenize, stem
from model import ChatbotNet

with open('intents.json', 'r') as f:
    intents = json.load(f)
#print(intents)    


all_words=[]
tags=[]
together=[]
for intent in intents['intents']:
    tag=intent['tag']
    #add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        together.append((w, tag))

ignore_words=['?','.','!',',',';','/']
all_words=[stem(w) for w in all_words if w not in ignore_words]
all_words=sorted(set(all_words))
tags=sorted(set(tags))


X_train=[]
y_train=[]
for (pattern_sentence,tag) in together:
    bag=bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label=tags.index(tag)
    y_train.append(label)

X_train=np.array(X_train)
y_train=np.array(y_train)

# Hyper-parameters 
num_epochs=1000
batch_size=8
learning_rate=0.0001
input_size=len(X_train[0])
hidden_size=32
output_size=len(tags)
print(input_size,output_size)

class CharDataset(Dataset):

    def __init__(self):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]


    def __len__(self):
        return self.n_samples

dataset=CharDataset()
train_loader=DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model=ChatbotNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words,labels) in train_loader:
        words=words.to(device)
        labels=torch.tensor(labels, dtype=torch.long, device=device)
        
        # Forward pass
        outputs=model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss=criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100==0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')


print(f'final loss: {loss.item():6f}')

data={
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

file="trained.pth"
torch.save(data, file)

print(f'training complete. file saved to {file}')