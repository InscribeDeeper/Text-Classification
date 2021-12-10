import re
from collections import Counter
import pandas as pd
import torch
from sklearn.model_selection import train_test_split  # , StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
##############################################################################################################
# Training option 1
##############################################################################################################
def BERT_data_loader(sentences_encoding, input_ids, one_hot_labels, batch_size=16, random_state=1234, test_size=0.1, testing=False):
    # Split
    one_hot_labels = torch.tensor(one_hot_labels, dtype=torch.int32)

    if not testing:
        train_inputs, validation_inputs, train_ids, validation_ids, train_labels, validation_labels = train_test_split(sentences_encoding, input_ids, one_hot_labels, random_state=random_state, test_size=test_size)

        # Convert all inputs and labels into torch tensors, the required datatype for our model.
        train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
        validation_inputs = torch.tensor(validation_inputs, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.int32)
        validation_labels = torch.tensor(validation_labels, dtype=torch.int32)

        # Create the DataLoader for our training set.
        train_data = TensorDataset(train_inputs, train_ids, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        # Create the DataLoader for our validation set.
        validation_data = TensorDataset(validation_inputs, validation_ids, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

        return train_dataloader, validation_dataloader

    else:
        batch_size = sentences_encoding.shape[0] if batch_size is None else batch_size
        test_inputs = torch.tensor(sentences_encoding, dtype=torch.float32)
        test_labels = torch.tensor(one_hot_labels, dtype=torch.int32)
        test_data = TensorDataset(test_inputs, input_ids, test_labels)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_size)
        return test_dataloader, None
