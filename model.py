from torch.nn import *
from torch import optim
import torch
import numpy as np


class Model(Module):
    def __init__(self, lr=0.001, n_epochs=5):
        super(Model, self).__init__()

        self.lr = lr
        self.epochs = n_epochs


        ############################ Your Code Here ############################
        # Initialize your model in this space
        # You can add arguements to the initialization as needed
        ########################################################################

        self.l1 = Sequential(
                    Linear(31, 28),
                    ReLU()
                )

        self.l2 = Sequential(
                    Linear(28, 1),
                    Sigmoid()
                )

        self.test = Sequential(
                Linear(28, 1),
                Sigmoid()
                )


    def forward(self, X):
        x = self.l1(X)
        return self.l2(x)

        # return self.test(X)


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

                loss = self.loss(pred.reshape(-1), y.float())

                loss.backward()

                self.optim.step()
                
                train_losses.append(loss.item())

                _, classes = torch.max(pred, dim=-1)
                correct += torch.sum(classes == y).item()
                total += x.size(0)

            train_accuracy = correct / total
            train_loss = np.mean(train_losses)

            total = 0
            correct = 0
            # for (x, y) in val_dl:
            #     pred = self.forward(x)

            #     loss = self.loss(pred, y)
            #     val_losses.append(loss.item())

            #     self.optim.step()

            #     _, classes = torch.max(pred, dim=1)
            #     correct += torch.sum(classes == y).item()
            #     total += x.size(0)
            # val_loss = np.mean(val_losses)


                  
            # Print progress
            if val_accuracy is not None:
                print("Epoch {}/{}, train_loss: {:.4f}, val_loss: {:.4f}, train_accuracy: {:.4f}, val_accuracy: {:.4f}"
                          .format(epoch+1, self.epochs, train_loss, val_loss, train_accuracy, val_accuracy))
            else:
                print("Epoch {}/{}, train_loss: {:.4f}, train_accuracy: {:.4f}"
                          .format(epoch+1, self.epochs, train_loss, train_accuracy)) 
        pass
            



    def predict_proba(self, x):
        ############################ Your Code Here ############################
        # Predict the probability of in-hospital mortaility for each x
        pass
        ########################################################################
        #return preds
