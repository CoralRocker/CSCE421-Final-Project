from torch.nn import *
from torch import optim
import torch
import numpy as np

from data import get_default_device, to_device

class Model(Module):
    def __init__(self, lr=0.001, n_epochs=5):
        super(Model, self).__init__()

        self.lr = lr
        self.epochs = n_epochs


        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        ########################################################################

        self.layer = Sequential(
                Linear(11, 11),
                ReLU(),
                Linear(11, 1),
                Sigmoid()
                )



    def forward(self, X):
        return self.layer(X)


    def fit(self, train_dl, val_dl):
        ############################ Your Code Here ############################
        # Fit your model to the training data here
        ########################################################################

        self.optim = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = MSELoss() 


        for epoch in range(self.epochs):

            val_accuracy = None
            train_accuracy = None
            total = 0
            correct = 0

            train_losses = []
            val_losses = []
            train_loss = 0
            val_loss = 0

            for i, (x, y) in enumerate(train_dl):
                pred = self(x)

                self.optim.zero_grad()

                loss = self.loss(pred, y.float())

                loss.backward()

                self.optim.step()
                
                train_losses.append(loss.item())
                # _, classes = torch.max(pred, dim=1)
                correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += pred.size(0)

            train_accuracy = correct / total
            train_loss = np.mean(train_losses)

            total = 0
            correct = 0

            for (x, y) in val_dl:
                pred = self.forward(x)

                loss = self.loss(pred, y.float())
                val_losses.append(loss.item())

                self.optim.step()

                # _, classes = torch.max(pred, dim=1)
                # correct += torch.sum(classes == y).item()
                correct += torch.sum(torch.round(pred).squeeze() == y).item()
                total += x.size(0)

            val_loss = np.mean(val_losses)
            val_accuracy = correct / total


                  
            # Print progress
            if val_accuracy is not None:
                print("\rEpoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_accuracy: {:.4f}, val_accuracy: {:.4f}"
                          .format(epoch+1, self.epochs, train_loss, val_loss, train_accuracy, val_accuracy), end="")
            else:
                print("\rEpoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.4f}"
                          .format(epoch+1, self.epochs, train_loss, train_accuracy), end='')
        pass
            



    def predict_proba(self, X):
        for x in X:
            print(self.forward(x))

