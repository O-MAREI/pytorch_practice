import csv
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Preparing the datasets
X_list = []
y_list = []
with open('delivery.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if row[0] != 'sepal_length':
            X_list.append(row[0:4])
            y_list.append(row[4])

X = torch.from_numpy(np.array((np.float_(X_list)))).type(torch.float)
le = preprocessing.LabelEncoder()
y_num = le.fit_transform(y_list)
y = torch.Tensor(y_num)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=42) # make the random split reproducible with a seed

# Setting the device based on availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# The Model
class LocationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=4, out_features=8),
            nn.ReLU(), 
            nn.Linear(in_features=8, out_features=8),
            nn.ReLU(), 
            nn.Linear(in_features=8, out_features=3), 
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

# A function for calculating accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

#----------------------- Model Creation and Training -----------------------

model = LocationModel().to(device)

# Loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)

# Training and testing loop:
torch.manual_seed(42)

# Number of epochs
epochs = 200

X_train, y_train = X_train.to(device), y_train.type(torch.LongTensor).to(device)
X_test, y_test = X_test.to(device), y_test.type(torch.LongTensor).to(device)

for epoch in range(epochs):
    # Training
    model.train

    # Forward pass
    y_logits = model(X_train)
    y_pred = torch.softmax(y_logits, dim = 1).argmax(dim = 1)

    # Calculate loss and accuracy
    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_true = y_train, y_pred = y_pred)

    # Optimizer
    optimizer.zero_grad()
    loss.backward()

    # Optimizer step
    optimizer.step()

    # Testing
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test)
        test_pred = torch.softmax(test_logits, dim = 1).argmax(dim = 1)

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true = y_test,
                               y_pred = test_pred)

#    Uncomment if you wish to see test and training accuracies across epochs
#    if epoch % 10 == 0:
#        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

#----------------------- Predictions and Plots -----------------------

# Uncomment to display a plot of the features and the predicted classification
'''
y_logits = model(X_test.to(device))
y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
df = pd.DataFrame(X_test.to("cpu"), columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
df['Class'] = y_pred.to("cpu")
sns.pairplot(df, hue='Class', palette='deep')
plt.show()
'''

# The final prediction:
current_batch = torch.Tensor([5.9, 3, 5.1, 1.8])
y_logit = model(current_batch.to(device))
y_prediction = torch.softmax(y_logit, dim=0).argmax(dim=0)
dict = {0: "A", 1: "B", 2: "C"}
print("The most likely upcoming delivery location is: {}".format(dict[y_prediction.item()]))

