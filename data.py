import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize

##
## The Following Code is Taken from Homework 6
## All credit goes to Dr. Mortazavi
##
def get_default_device():
    """Use GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for batch in self.dl:
            yield to_device(batch, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dl)
##
## End of Copied Code Section
##


def load_data(x_path):
    return pd.read_csv(x_path, low_memory=False)

###
### Get the indeces for the validation and training splits, respectively
###
### n is the size of the dataset
### split is the fraction of the data to use for training
### random_state is the seed for the rng
def get_data_indeces(n, split=0.8, random_state=42):
    nval = int(n * split) 
    np.random.seed(random_state)

    idxs = np.random.permutation(n)

    # Return Small Split, Large Split
    return idxs[nval:], idxs[:nval]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.data[idx]

        if self.transform: 
            image = self.transform(image)

        return image, label
###
### Get the training and validation dataloaders as well as the chosen device
###
### x is the non-preprocessed data
### y is the non-preprocessed labels
### split is the fraction of the data to use for training
### random_state is the seed for the rng
def get_dataloaders(x, y, split=0.8, random_state=42):

    batch_size = 2

    p = preprocess_x(x) # Get the preprocessed X data

    # Sort by patient ID
    p.sort_values(by='patientunitstayid', inplace=True)

    # Sorty Y by patient ID
    y.sort_values(by='patientunitstayid', inplace=True)

    # Get default device
    device = get_default_device()

    # Create the dataset of tensors and labels
    dataset = CustomDataset(p.to_numpy('float32'),
                            y['hospitaldischargestatus'].to_numpy(),
                            torch.tensor)
    # dataset = np.array(tuple(zip(
    #     torch.tensor(p.to_numpy()), 
    #     y['hospitaldischargestatus'].to_numpy()
    #     )), dtype='float32')
    # print(f"Dataset Shape: {dataset.shape}")
    # print(f"Dataset: {dataset.dtype}")

    val_idx, train_idx = get_data_indeces(p.shape[0], split, random_state)

    train_sampler = SubsetRandomSampler(train_idx)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_idx)
    val_dl = DataLoader(dataset, batch_size, sampler=val_sampler)

    print(f"Validation index length: {len(val_idx)}, Validation DL length: {len(val_dl)}")
    print(f"Training index length: {len(train_idx)}, Training DL length: {len(train_dl)}")

    print(f"Chosen device: {device}")

    return DeviceDataLoader(train_dl, device), DeviceDataLoader(val_dl, device), device


def split_data(x, y, split=0.8, random_state=42):
    # This assumes that X has been preprocessed and its size normalized to be the
    # same as Y


    nval_train = int(x.shape[0] * split)
    print(f"X: {x.shape}, Y: {y.shape}")

    np.random.seed(random_state)

    idxs = np.random.permutation(x.shape[0] - 1)

    idx_train = idxs[:nval_train]

    idx_test = idxs[nval_train:]

    return x.iloc[idx_train], y.iloc[idx_train], x.iloc[idx_test], y.iloc[idx_test]


def preprocess_x(df):

    ### Aggregate the all the rows of a certain patient
    patientcounts = df.groupby('patientunitstayid')[['patientunitstayid']]\
            .transform('count')
    df['patient row count'] = patientcounts # Each row holds how many times the patient was measured

    ### Create dummies for categorical columns
    df = pd.get_dummies(df, columns= ['ethnicity', 'gender', 'celllabel', 'labmeasurenamesystem'])


    ### Looking at relevant columns (not sure if offset is relevant, currently ignoring it)
    nursing_cols = df[['patientunitstayid', 'nursingchartcelltypevalname', 'nursingchartvalue']]
    ### Some values are "Unable to score due to medication" this just drops those from the table (could probably be dealt with better)
    nursing_cols['nursingchartvalue'] = pd.to_numeric(nursing_cols['nursingchartvalue'], errors='coerce')  # sets non-numeric cells to NaN which then get dropped
    nursing_cols.dropna(inplace=True)

    nursing_cols = nursing_cols.groupby(['patientunitstayid', 'nursingchartcelltypevalname']) \
                            .mean() \
                            .reset_index()

    ### Creates a column for each unique nursingchartcelltypevalname, fills in values from mean gotten above 
    ### TODO: Is the mean a good metric to use here?
    nursing_cols = nursing_cols.pivot(index='patientunitstayid', columns='nursingchartcelltypevalname', values='nursingchartvalue')
    nursing_cols.columns = [str(col) for col in nursing_cols.columns]

    ### Merge back into df, drop og columns
    df = pd.merge(df, nursing_cols, on='patientunitstayid', how='left')
    df = df.drop(columns=['nursingchartcelltypevalname', 'nursingchartvalue'])


    ### Looking at relevant columns (not sure if offset is relevant, currently ignoring it)
    ### NOT currently considering the units of measurement. Naively assumes they're all the same per labname
    lab_cols = df[['patientunitstayid', 'labname', 'labresult']]
    ### Some values are "Unable to score due to medication" this just drops those from the table (could probably be dealt with better)
    lab_cols['labresult'] = pd.to_numeric(lab_cols['labresult'], errors='coerce')  # sets non-numeric cells to NaN which then get dropped
    lab_cols.dropna(inplace=True)

    lab_cols = lab_cols.groupby(['patientunitstayid', 'labname']) \
                            .mean() \
                            .reset_index()

    ### Creates a column for each unique labname, fills in values from mean gotten above 
    ### TODO: Is the mean a good metric to use here?
    lab_cols = lab_cols.pivot(index='patientunitstayid', columns='labname', values='labresult')
    lab_cols.columns = [str(col) for col in lab_cols.columns]

    ### Merge back into df, drop og columns
    df = pd.merge(df, lab_cols, on='patientunitstayid', how='left')
    df = df.drop(columns=['labname', 'labresult'])


    df.drop('Unnamed: 0', axis=1, inplace=True)

    ### TODO: Revisit this
    # This drops all the duplicate patient stays past the first one
    df.drop_duplicates('patientunitstayid', inplace=True)

    ### Change Ages from string to int
    ## Set the age "> 89" to 90
    df['age'].loc[df['age'] == '> 89'] = '90'
    df['age'] = df['age'].astype('float32')

    ### Change CellAttributeValue to float32
    # # df['cellattributevalue'] = df['cellattributevalue'].astype('float32')
    # # df['nursingchartvalue'] = df['nursingchartvalue'].astype('float32')
    # 
    # df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(['offset', 'labresult'], axis=1, inplace=True)
    df['cellattributevalue'] = df['cellattributevalue'].astype('float32')
    # df['nursingchartvalue'] = df['nursingchartvalue'].astype('float32')

    # df.fillna(0, inplace=True)

    
    return df
    
