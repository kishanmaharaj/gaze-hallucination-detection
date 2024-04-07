import os
import torch


model_dict = {
    "albert-base-v2": 768,
    "albert-large-v2": 1024,
    "albert-xlarge-v2": 2048,
    "albert-xxlarge-v2": 4096,
    "bert-base-uncased": 768,
    "bert-large-uncased": 1024,
    "microsoft/deberta-v3-base": 768,
    "cross-encoder/nli-deberta-v3-base": 768,
    "sentence-transformers/nli-bert-base": 768,
    "roberta-base": 768, 
}

class_1_models = [
                  "albert-large-v2",
                  "albert-xlarge-v2",
                  "albert-xxlarge-v2",
                  "bert-base-uncased",
                  "bert-large-uncased"]

class_2_models = [
    "albert-base-v2",
    "microsoft/deberta-v3-base",
     "cross-encoder/nli-deberta-v3-base", 
     "sentence-transformers/nli-bert-base"
]


# Fixation Network
fx_batch_size = 8
gaze_seq_len = 150 # Sequence length of gaze data
fix_dropout = 0.5
fix_hidden_dim = 128


# Hallucination Model
freeze_bert = False  # if True, freeze the encoder weights and only update the classification layer weights
maxlen = 320  # maximum length of the tokenized input sentence pair : if greater than "maxlen", the input is truncated and else if smaller, the input is padded
batch_size = 128  # batch size
iters_to_accumulate = 2  # the gradient accumulation adds gradients over an effective batch of size : bs * iters_to_accumulate. If set to "1", you get the usual batch size
lr = 2e-5  # learning rate
epochs = 3  # number of training epochs