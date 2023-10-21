import torch
import numpy as np
from transformers import BertTokenizer
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from flask_cors import CORS
from flask import Flask, jsonify, request
from torch.utils.data import DataLoader
tokenizer = AutoTokenizer.from_pretrained(r'/data03/dengwentao-slurm/Legal_LLM/Library/model/Lawformer')
labels = {'business':0,
          'entertainment':1,
          'sport':2,
          'tech':3,
          'politics':4
          }

class Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.texts = [tokenizer(text,
                                padding='max_length',
                                max_length=512,
                                truncation=True,
                                return_tensors="pt")
                      for text in df['all']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.texts)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts


if __name__ == '__main__':
    filename = 'output.csv'
    df = pd.read_csv(filename)
    dataset = Dataset(df)
    print(dataset[0])
    data = DataLoader(dataset, batch_size=64)
    for i in data:
        print(i['input_ids'].shape)
        print(i['attention_mask'].shape)


