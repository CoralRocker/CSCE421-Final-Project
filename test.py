# from project_utils import create_data_for_project

# data = create_data_for_project(".")

import itertools

import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from torch import optim


from data import *
from parser import parse
from model import Model

np.seterr(all='raise')

def main():
    args = parse()

    device = get_default_device()


    model = torch.load("./trained_model_auc")    
    model = to_device(model, device)

    test_x = load_data("test_x.csv")

    x_test, ids = preprocess_test(test_x)

    x_tensor = to_device(torch.tensor(x_test.to_numpy('float32')), device)

    probs = model.predict_proba(x_tensor)

    submission = list(zip(ids, probs))
    with open("submission_test.csv", "w") as file:
        file.write("patientunitstayid,hospitaldischargestatus\n")
        for id, prob in submission:
            file.write(f"{int(id)},{prob :f}\n")
    

if __name__ == "__main__":
    main()
