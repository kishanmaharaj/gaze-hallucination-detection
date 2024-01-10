import nltk
# nltk.download('punkt')
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk.data
from sentence_transformers import SentenceTransformer, util
import pickle
from tqdm import tqdm
import sys
from collections import Counter

data_file = sys.argv[1]
output_file = sys.argv[2]

models = ['LaBSE', 'gtr-t5-large', 'all-roberta-large-v1', 'all-mpnet-base-v1', 'all-mpnet-base-v2']
data_df = pd.read_csv(data_file)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

most_voted_df = pd.DataFrame(columns=['instance_id','claim','LaBSE', 'gtr-t5-large', 'all-roberta-large-v1', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'most_voted_sent'])
instance_ids = []
sentence_ids = []
sentences = []
claims = []

inst=1

for ind in tqdm(data_df.index):    
    claim = str(data_df['claim'][ind])
    claims.append(claim)
    instance_ids.append(inst)
    inst+=1


most_voted_df['instance_id'] = instance_ids
most_voted_df['claim'] = claims

for model_name in models:
    model = SentenceTransformer(model_name)
    row = []
    
    for ind in tqdm(data_df.index):
        text = str(data_df['text'][ind])
        text_sents = tokenizer.tokenize(text)
        sent_embeddings = model.encode(text_sents)

        claim = str(data_df['claim'][ind])
        claim_embedding  = model.encode(claim)

        cosine_scores = []
        for sent in range(len(text_sents)):
            cos_score = util.pytorch_cos_sim(claim_embedding, sent_embeddings[sent])
            cosine_scores.append(cos_score)
        
        max_cos_score_sent = 0
        for sent in range(len(text_sents)):
            if cosine_scores[sent] > cosine_scores[max_cos_score_sent]:
                max_cos_score_sent = sent
        
        sentence = text_sents[max_cos_score_sent]
        row.append(sentence)

    most_voted_df[str(model_name)] = row

most_voted_sents = []
for ind in tqdm(most_voted_df.index):
    sents = []
    sents.append(str(most_voted_df['LaBSE'][ind]))
    sents.append(str(most_voted_df['gtr-t5-large'][ind]))
    sents.append(str(most_voted_df['all-roberta-large-v1'][ind]))
    sents.append(str(most_voted_df['all-mpnet-base-v1'][ind]))
    sents.append(str(most_voted_df['all-mpnet-base-v2'][ind]))

    most_voted_sent = (Counter(sents).most_common(1)[0][0])
    most_voted_sents.append(most_voted_sent)

most_voted_df['most_voted_sent'] = most_voted_sents

most_voted_df.to_csv(output_file)




