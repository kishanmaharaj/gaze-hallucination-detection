import config
from collections import OrderedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, BertModel

    


## Custom class for gaze data 
class GazeDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model='bert-base-uncased'):

        self.data = data  # pandas dataframe
        
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = str(self.data.loc[index, 'chunked_words'])


        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = np.array(self.data.loc[index, 'label'])

            pad_size = config.gaze_seq_len - len(label)
            label_padded = np.pad(label, (0, pad_size), 'constant', constant_values=(0)) 
            return token_ids, attn_masks, token_type_ids, label_padded
            
        else:
            return token_ids, attn_masks, token_type_ids


## Fixation Network
class FixNN(nn.Module):
    def __init__(
        self,
        dropout,
        hidden_dim,
        freeze_bert = False,
    ):
        super().__init__()
        prelayers = OrderedDict()
        postlayers = OrderedDict()
        
        embedding_dim = 768
        
        
        self.bert_layer = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True, output_attentions=True)
        
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
            print("BERT Frozen")

        postlayers["ff_layer"] = nn.Linear(embedding_dim, hidden_dim // 2)
        postlayers["ff_activation"] = nn.ReLU()
        postlayers["output_layer"] = nn.Linear(hidden_dim // 2, 1)


        self.pre = nn.Sequential(prelayers) 
        self.post = nn.Sequential(postlayers)

    def forward(self, x, attn_masks, token_type_ids):
        
        out_pre = self.bert_layer(x, attn_masks, token_type_ids)
        
        out_post = self.post(out_pre['last_hidden_state'])
        
        score = torch.sigmoid(out_post)
 
        return  score



