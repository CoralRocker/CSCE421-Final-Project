import pandas as pd
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize

from sklearn.preprocessing import StandardScaler

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
### Get a sample of x and y where there is an equal amount of both deaths and lives
###
### This uses sampling with replacement from a dataframe of dead and live people 
### To generate a new dataframe with an equal amount of both.
def getOversampledDataset(x, y, n):
    # Labels for Dead And Alive people
    dead = y.loc[y['hospitaldischargestatus'] == 1]
    live = y.loc[y['hospitaldischargestatus'] == 0]

    # Data of dead and alive people
    dead_ppl = x.loc[x['patientunitstayid'].isin(dead['patientunitstayid'])]
    live_ppl = x.loc[x['patientunitstayid'].isin(live['patientunitstayid'])]

    # Number of dead and living people
    n_dead = int(n / 2)
    n_live = n - n_dead

    # Combined Labels of dead and living people
    people_labels = pd.concat([dead.sample(n=n_dead, replace=True), live.sample(n=n_live, replace=True)])

    # Add each ID's data to the dataframe
    people_data = pd.DataFrame(columns=x.columns) 
    for id in people_labels['patientunitstayid']:
        people_data = pd.concat([
                people_data, 
                dead_ppl.loc[dead_ppl['patientunitstayid'] == id], 
                live_ppl.loc[live_ppl['patientunitstayid'] == id]
            ]) 

    # Sort by patient ID to align both living and dead
    people_labels.sort_values(by='patientunitstayid', inplace=True)
    people_data.sort_values(by='patientunitstayid', inplace=True)
    
    return people_data, people_labels
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
### batch_size is the batch size of the dataset
### sample_size is the number of datapoints to load. An oversampling method is used
### split is the fraction of the data to use for training
### random_state is the seed for the rng
def get_dataloaders(x, y, batch_size=1, sample_size=1000, split=0.8, random_state=42):

    # Get default device
    device = get_default_device()

    p = preprocess_x(x) # Get the preprocessed X data
    
    data = None
    labels = None
    ##
    ## Oversampled Data Process
    ##
    data, labels = getOversampledDataset(p, y, sample_size)
    data.drop('patientunitstayid', axis=1, inplace=True)

    dataset = CustomDataset(data.to_numpy('float32'),
                            labels['hospitaldischargestatus'].to_numpy(),
                            torch.tensor)

    ## No sampling data
    # Sort by patient ID
    # p.sort_values(by='patientunitstayid', inplace=True)
    # p.drop('patientunitstayid', axis=1, inplace=True)

    # # Sorty Y by patient ID
    # y.sort_values(by='patientunitstayid', inplace=True)

    # # Create the dataset of tensors and labels
    # dataset = CustomDataset(p.to_numpy('float32'),
    #                         y['hospitaldischargestatus'].to_numpy(),
    #                         torch.tensor)

    val_idx, train_idx = get_data_indeces(sample_size, split, random_state)

    train_sampler = SubsetRandomSampler(train_idx)
    train_dl = DataLoader(dataset, batch_size, sampler=train_sampler)

    val_sampler = SubsetRandomSampler(val_idx)
    val_dl = DataLoader(dataset, batch_size, sampler=val_sampler)

    # print(f"Validation index length: {len(val_idx)}, Validation DL length: {len(val_dl)}")
    # print(f"Training index length: {len(train_idx)}, Training DL length: {len(train_dl)}")

    # print(f"Chosen device: {device}")

    return DeviceDataLoader(train_dl, device), DeviceDataLoader(val_dl, device), device, data, labels


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

##
## Get preprocessed data for testing
##
## Returns the data, and the ids for the data
##
## If a Y is given, it is sorted by ID with the test data and returned as an array of labels
def preprocess_test(df, y = None):
    p = preprocess_x(df)

    ids = p['patientunitstayid'].to_numpy()
    
    if y is not None:
        p.sort_values(by='patientunitstayid', inplace=True)
        y.sort_values(by='patientunitstayid', inplace=True) 

    p.drop('patientunitstayid', axis=1, inplace=True)

    if y is not None:
        return p, ids, y['hospitaldischargestatus'].to_numpy('float32')

    return p, ids

def preprocess_x(df):
    scaler = StandardScaler() 

    df['unitvisitnumber'] = scaler.fit_transform(df['unitvisitnumber'].values.reshape(-1, 1))

    ### Aggregate the all the rows of a certain patient
    patientcounts = df.groupby('patientunitstayid')[['patientunitstayid']]\
            .transform('count')
    df['patient row count'] = patientcounts # Each row holds how many times the patient was measured
    df['patient row count'] = scaler.fit_transform(df['patient row count'].values.reshape(-1, 1))

    ### Create dummies for categorical columns
    df = pd.get_dummies(df, columns= ['ethnicity', 'gender', 'cellattributevalue', 'celllabel', 'labmeasurenamesystem'])

    ### Max offset
    max_offset = df.groupby('patientunitstayid')['offset'] \
                    .max() \
                    .reset_index(name='max_offset')
    df = pd.merge(df, max_offset, on='patientunitstayid', how='left')
    df['max_offset'] = scaler.fit_transform(df['max_offset'].values.reshape(-1, 1))

    ### Median offset
    median_offset = df.groupby('patientunitstayid')['offset'] \
                    .median() \
                    .reset_index(name='median_offset')
    df = pd.merge(df, median_offset, on='patientunitstayid', how='left')
    df['median_offset'] = scaler.fit_transform(df['median_offset'].values.reshape(-1, 1))


    ### Capillary Refill
    cell_cols = df.loc[:, ['patientunitstayid', 'celllabel_Capillary Refill']]
    cell_cols = cell_cols.groupby(['patientunitstayid', 'celllabel_Capillary Refill']) \
                            .mean() \
                            .reset_index()
    cell_cols = cell_cols.pivot(index='patientunitstayid', columns='celllabel_Capillary Refill', values='celllabel_Capillary Refill')
    cell_cols.rename(columns={1:'Had Capillary Refill'}, inplace=True)
    cell_cols.fillna(0, inplace=True)


    ### Merge back into df, drop og columns
    df = pd.merge(df, cell_cols, on='patientunitstayid', how='left')
    df = df.drop(columns=[0, 'celllabel_Capillary Refill'])

    ### Looking at relevant columns (not sure if offset is relevant, currently ignoring it)
    nursing_cols = df.loc[:, ['patientunitstayid', 'nursingchartcelltypevalname', 'nursingchartvalue']]
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
    df = df.drop(columns=['nursingchartvalue'])
    df = pd.get_dummies(df, columns=['nursingchartcelltypevalname'])

    df['GCS Total'] = scaler.fit_transform(df['GCS Total'].values.reshape(-1, 1))
    df['Heart Rate'] = scaler.fit_transform(df['Heart Rate'].values.reshape(-1, 1))

    ### Grouped these together because they seemed sufficiently related
    df[['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic']] = scaler.fit_transform(\
        df[['Invasive BP Diastolic', 'Invasive BP Mean', 'Invasive BP Systolic']].values)
    df[['Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic']] = scaler.fit_transform(\
        df[['Non-Invasive BP Diastolic', 'Non-Invasive BP Mean', 'Non-Invasive BP Systolic']].values)
    
    df['O2 Saturation'] = scaler.fit_transform(df['O2 Saturation'].values.reshape(-1, 1))
    df['Respiratory Rate'] = scaler.fit_transform(df['Respiratory Rate'].values.reshape(-1, 1))

    # df['nursingchartcelltypevalname_GCS Total'] = (df['GCS Total'].notnull()).astype(int)
    
    df.loc[df['GCS Total'].notnull(), 'nursingchartcelltypevalname_GCS Total'] = 1
    df.loc[df['Heart Rate'].notnull(), 'nursingchartcelltypevalname_Heart Rate'] = 1
    df.loc[df['Invasive BP Diastolic'].notnull(), 'nursingchartcelltypevalname_Invasive BP Diastolic'] = 1
    df.loc[df['Invasive BP Mean'].notnull(), 'nursingchartcelltypevalname_Invasive BP Mean'] = 1
    df.loc[df['Invasive BP Systolic'].notnull(), 'nursingchartcelltypevalname_Invasive BP Systolic'] = 1
    df.loc[df['Non-Invasive BP Diastolic'].notnull(), 'nursingchartcelltypevalname_Non-Invasive BP Diastolic'] = 1
    df.loc[df['Non-Invasive BP Mean'].notnull(), 'nursingchartcelltypevalname_Non-Invasive BP Mean'] = 1
    df.loc[df['Non-Invasive BP Systolic'].notnull(), 'nursingchartcelltypevalname_Non-Invasive BP Systolic'] = 1
    df.loc[df['O2 Saturation'].notnull(), 'nursingchartcelltypevalname_O2 Saturation'] = 1
    df.loc[df['Respiratory Rate'].notnull(), 'nursingchartcelltypevalname_Respiratory Rate'] = 1

    ### Looking at relevant columns (not sure if offset is relevant, currently ignoring it)
    ### NOT currently considering the units of measurement. Naively assumes they're all the same per labname
    lab_cols = df.loc[:, ['patientunitstayid', 'labname', 'labresult']]
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
    df = pd.get_dummies(df, columns=['labname'])
    df = df.drop(columns=['labresult'])
    
    df['glucose'] = scaler.fit_transform(df['glucose'].values.reshape(-1, 1))
    df['pH'] = scaler.fit_transform(df['pH'].values.reshape(-1, 1))

    df.loc[df['glucose'].notnull(), 'labname_glucose'] = 1 
    df.loc[df['pH'].notnull(), 'labname_pH'] = 1
    df['glucose'].fillna(0, inplace=True)
    df['pH'].fillna(0, inplace=True)

    # # Gaultier Old Code
    # df = df.drop(columns=['nursingchartcelltypevalname', 'nursingchartvalue', 
    #                       'celllabel', 'labmeasurenamesystem', 
    #                       'labname', 'cellattributevalue',
    #                       'unitvisitnumber', 'patient row count'],
    #                         axis=1)
    # 

    # ### Create dummies for categorical columns
    # df = pd.get_dummies(df, columns= ['ethnicity', 'gender'])

    df.drop('Unnamed: 0', axis=1, inplace=True)

    ### TODO: Revisit this
    # This drops all the duplicate patient stays past the first one
    df.drop_duplicates('patientunitstayid', inplace=True)

    ## Set NaNs to mean of column
    # df['age'].fillna(df['age'].mean(), inplace=True)
    # df['admissionweight'].fillna(df['admissionweight'].mean(), inplace=True)
    # df['admissionheight'].fillna(df['admissionheight'].mean(), inplace=True)
    ## Set NaNs to 0
    # df['admissionweight'].fillna(0, inplace=True)
    # df['admissionheight'].fillna(0, inplace=True)

    df.insert(0, 'Has Weight', 0)
    df.insert(0, 'Has Height', 0)

    df['admissionweight'] = scaler.fit_transform(df['admissionweight'].values.reshape(-1, 1))
    df['admissionheight'] = scaler.fit_transform(df['admissionheight'].values.reshape(-1, 1))

    df.loc[df['admissionweight'].notnull(), 'Has Weight'] = 1
    df.loc[df['admissionheight'].notnull(), 'Has Height'] = 1
    
    df['admissionweight'].fillna(0, inplace=True)
    df['admissionheight'].fillna(0, inplace=True)

    df['GCS Total'].fillna(0, inplace=True)
    df['Heart Rate'].fillna(0, inplace=True)
    df['Invasive BP Diastolic'].fillna(0, inplace=True)
    df['Invasive BP Mean'].fillna(0, inplace=True)
    df['Invasive BP Systolic'].fillna(0, inplace=True)
    df['Non-Invasive BP Diastolic'].fillna(0, inplace=True)
    df['Non-Invasive BP Mean'].fillna(0, inplace=True)
    df['Non-Invasive BP Systolic'].fillna(0, inplace=True)
    df['O2 Saturation'].fillna(0, inplace=True)
    df['Respiratory Rate'].fillna(0, inplace=True)

    ### Change Ages from string to int
    ## Set the age "> 89" to 90
    df.insert(0, 'Has Age', 0)
    df.loc[df['age'].notna(), 'Has Age'] = 1
    df['age'].fillna('0', inplace=True)

    agemask = df['age'].str.fullmatch(r'> 89')
    df.loc[agemask, 'age'] = '90'
    df['age'] = df['age'].astype('float32')

    # df.drop('Unnamed: 0', axis=1, inplace=True)
    df.drop(['offset'], axis=1, inplace=True)
    # df['cellattributevalue'] = df['cellattributevalue'].astype('float32')
    # df['nursingchartvalue'] = df['nursingchartvalue'].astype('float32')


    ## Ensure All One-Hot Encodings are accounted for
    encodings = ['ethnicity_African American', 'ethnicity_Asian', 'ethnicity_Caucasian',
                 'ethnicity_Hispanic', 'ethnicity_Native American',
                 'ethnicity_Other/Unknown', 'gender_Female', 'gender_Male']

    for encoding in encodings:
        if encoding not in list(df.columns):
            df.insert(0, encoding, 0)

    df.to_csv('data.csv', index=False)
    
    return df
    
