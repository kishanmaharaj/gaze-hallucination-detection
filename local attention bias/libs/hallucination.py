import config
from .fixation_pred import FixNN
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


class hallucination_dataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, language_model='bert-base-uncased'):

        self.data = data  # pandas dataframe
        #Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(language_model)  

        self.maxlen = maxlen
        self.with_labels = with_labels 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent1, sent2, 
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,  
                                      return_tensors='pt')  # Return torch.Tensor objects
        
        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.data.loc[index, 'label']
            return token_ids, attn_masks, token_type_ids, label  
        else:
            return token_ids, attn_masks, token_type_ids



class hallucination_classifier(nn.Module):

    def __init__(self, language_model="bert-base-uncased", fix_active=False, maxlen=320, checkpoint_tsm=None,  freeze_bert=False, tsm_active=True, infer=False):
        super(hallucination_classifier, self).__init__()

        self.sequence_len = maxlen

        self.tsm_active = tsm_active

        self.lm_name = language_model

        self.fix_active = fix_active
        self.infer = infer
        
        #  Instantiating  model object
        self.language_model = AutoModel.from_pretrained(self.lm_name, torch_dtype=torch.float32)

        
        #  Fix the hidden-state size of the encoder outputs 
        hidden_size = config.model_dict[self.lm_name]   # model_dict.get(bert_model)


        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        if self.tsm_active:
            if self.fix_active:
                self.tsm_layer = FixNN(dropout=config.fix_dropout, hidden_dim=config.fix_hidden_dim)
                self.tsm_layer.load_state_dict(torch.load(checkpoint_tsm))
                print("Fixation Network Initialized with checkpoint")

            else:
                self.tsm_layer = FixNN(dropout=config.fix_dropout, hidden_dim=config.fix_hidden_dim)
                print("Fixation Network Initialized randomly")

            self.cls_layer = nn.Linear(hidden_size*self.sequence_len, 1)
            
            print("TSM Active")

        elif language_model in config.class_1_models:
            self.cls_layer = nn.Linear(hidden_size, 1)
        
        else:
            self.cls_layer = nn.Linear(hidden_size*self.sequence_len, 1)
            
        self.dropout = nn.Dropout(p=0.1)

    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids: Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        # pooler_outpo = self.bert_layer(input_ids, attn_masks, token_type_ids, return_dict=True)
        cont_reps = None 
        logits = None
        
        if self.lm_name in config.class_1_models:
            cont_reps, pooler_output = self.language_model(input_ids, attn_masks, token_type_ids, return_dict=False)
        
        elif self.lm_name in config.class_2_models:
            cont_reps = self.language_model(input_ids, attn_masks, token_type_ids, return_dict=False)[0]


        if self.tsm_active:
            
            
            tsm_output = self.tsm_layer(input_ids, attn_masks, token_type_ids) 
            current_batch, curent_seq_len, curent_hidden_size = cont_reps.size()
            logits = self.cls_layer(self.dropout(cont_reps*tsm_output).view(current_batch, curent_hidden_size*curent_seq_len))
        
        else:
            current_batch, curent_seq_len, curent_hidden_size = cont_reps.size()

            if self.lm_name in config.class_1_models:
                logits = self.cls_layer(self.dropout(pooler_output))

            elif self.lm_name in config.class_2_models:
                logits = self.cls_layer(self.dropout(cont_reps).view(current_batch, curent_hidden_size*curent_seq_len))

        if self.infer and self.tsm_active:
            return logits, tsm_output

        else:
            return logits



