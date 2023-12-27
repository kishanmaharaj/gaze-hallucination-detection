import ast
import config
import numpy as np 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def min_max_scale(lst):
    """For a single list: used in IITB Gaze hallucination data"""
    
    min_val = min(lst)
    max_val = max(lst)
    if min_val == max_val:
        # Handle division by zero when min and max values are the same
        scaled_list = [0.0] * len(lst)
    else:
        scaled_list = [(x - min_val) / (max_val - min_val) for x in lst]
      
    return scaled_list


def get_iitb_gaze_hallucination():
    
    selected_columns = [ 'claim_words', 'claim_fixation', 'final_sent_words', 'final_sent_fixation', 'label', 'source_set']
    
    text = []
    fixations = []
    source_set = []
    
    for i in range(1, 6):
        current_subject = pd.read_csv("data/gaze_data/iitb-hallucination-data/all_instances/tsm_training_dat_sub_labeled"+str(i)+"_500.csv")[selected_columns]
        
        claims = [ i[1:] for i in map(ast.literal_eval, current_subject['claim_words'])]
        final_sent = [ i[1:] for i in map(ast.literal_eval, current_subject['final_sent_words'])]
        
        
        claim_fixations = [min_max_scale(list(map(ast.literal_eval, i.strip("[]").split(', ')))[1:]) for i in current_subject['claim_fixation']]
        final_sent_fixations = [min_max_scale(list(map(ast.literal_eval, i.strip("[]").split(', ')))[1:]) for i in current_subject['final_sent_fixation']]
    
        current_source_set = list(current_subject['source_set'])
        source_set += current_source_set
        
        for i, j in zip(claims, final_sent):
            text.append(i+j)
            
        
        for i, j in zip(claim_fixations, final_sent_fixations):
            fixations.append(i+j)
    
    
    df_iitb = pd.DataFrame()
    df_iitb['chunked_words'] = text
    df_iitb['label'] = fixations
    df_iitb['source_set'] = source_set
    
    
    df_iitb_train = df_iitb[df_iitb['source_set']=="Train"][['chunked_words', 'label']].reset_index(drop=True)
    df_iitb_test = df_iitb[df_iitb['source_set']=="test"][['chunked_words', 'label']].reset_index(drop=True)
    df_iitb_val = df_iitb[df_iitb['source_set']=="dev"][['chunked_words', 'label']].reset_index(drop=True)

    

    return df_iitb_train, df_iitb_test, df_iitb_val


def preprocessing_provo():
    ## Provo Corpus
    provo_data = pd.read_csv("data/gaze_data/provo/Provo_Corpus-Eyetracking_Data.csv")
    
    # selecting relevant columns
    selected_columns = ["TRIAL_INDEX", "Word_Cleaned", "Word_POS", "IA_DWELL_TIME"]
    
    provo_data = provo_data[selected_columns]
    
    # print("Null Values")
    # print("Words: ", sum(provo_data['Word_Cleaned'].isna()))
    # print("TRIAL_INDEX: ", sum(provo_data['TRIAL_INDEX'].isna()))
    # print("Word_POS: ", sum(provo_data['Word_POS'].isna()))
    # print("IA_DWELL_TIME: ", sum(provo_data['IA_DWELL_TIME'].isna()))
    
    # Remove the rows with empty words
    provo_data = provo_data[provo_data['Word_Cleaned'].notna()]
    
    # Fill empty tags with unknown 
    provo_data = provo_data.fillna('X')
    
    
    # select the required gaze feature for the model
    ## IA_DWELL_TIME: Dwell time of Intrest Area
    gaze_feature = "IA_DWELL_TIME"

    
    # Trial Wise Scaling 
    ## PROVO corpus is has gaze features ordered by TRIAL_INDEX (The number/ID of the trial).
    ## We need to normalize the data, such that each trial has IA_DWELL_TIME values ranging from 0 to 1.
    ## Current values of IA_DWELL_TIME are very low and can result in poor learning of the model. 
    scaler = MinMaxScaler()
    
    max_trail = max(provo_data['TRIAL_INDEX'])
    min_trail = min(provo_data['TRIAL_INDEX'])  
    scaled_fixation = []
    
    for i in range(min_trail, max_trail+1):
        
        current_trail_fixation = np.array(provo_data[provo_data['TRIAL_INDEX']==i]["IA_DWELL_TIME"]).reshape(-1, 1)
        
        scaled = scaler.fit_transform(current_trail_fixation)
        cur_fix = scaled.reshape(len(scaled)).tolist()
        scaled_fixation = scaled_fixation + cur_fix
    
        
    # All Words and word level fixation values
    all_words = list(provo_data["Word_Cleaned"])
    fixation = scaled_fixation #list(geco_data[gaze_feature])
    pos_tags = list(provo_data["Word_POS"])
    
    
    # Preprocessing words: 1) Lower casing 2) Removing punctuations
    all_words = list(map(str.lower,all_words))
    all_words_proc = []
    for i in all_words:
        all_words_proc.append(re.sub(r'[^\w\s]', '', i))
        
    # Output should be equal
    print("Total Words: ", len(all_words_proc), "\nTotal Fixation points: ", len(all_words))
    
    
    seq_len = config.gaze_seq_len
    
    chunks_words = [all_words_proc[x:x+seq_len] for x in range(0, len(all_words_proc), seq_len)]
    chunks_fixation = [fixation[x:x+seq_len] for x in range(0, len(fixation), seq_len)]
    
    chunks_pos_tags = [pos_tags[x:x+seq_len] for x in range(0, len(pos_tags), seq_len)]

    return chunks_words, chunks_fixation, chunks_pos_tags


def preprocessing_geco():
    geco_data =  pd.read_csv("data/gaze_data/geco/selected_columns.csv") 
    
    
    # select the required gaze feature for the model
    ## WORD_FIXATION_%: Percentage of all fixations in a trial falling on the current word.
    gaze_feature = "WORD_FIXATION_%"
    
    
    # Trial Wise Scaling 
    
    ## GECO corpus is has gaze features ordered by TRIAL number (The number/ID of the trial).
    ## We need to normalize the data, such that each trial has WORD_FIXATION_% values ranging from 0 to 1.
    ## Current values of WORD_FIXATION_% are very low and can result in poor learning of the model. 
    scaler = MinMaxScaler()
    
    max_trail = max(geco_data['TRIAL'])
    min_trail = min(geco_data['TRIAL'])  
    scaled_fixation = []
    
    for i in range(min_trail, max_trail+1):
        
        current_trail_fixation = np.array(geco_data[geco_data['TRIAL']==i]["WORD_FIXATION_%"]).reshape(-1, 1)
        
        scaled = scaler.fit_transform(current_trail_fixation)
        cur_fix = scaled.reshape(len(scaled)).tolist()
        scaled_fixation = scaled_fixation + cur_fix
    
        
    # All Words and word level fixation values
    all_words = list(geco_data["WORD"])
    fixation = scaled_fixation #list(geco_data[gaze_feature])
    
    # Preprocessing words: 1) Lower casing 2) Removing punctuations
    all_words = list(map(str.lower,all_words))
    all_words_proc = []
    for i in all_words:
        all_words_proc.append(re.sub(r'[^\w\s]', '', i))
        
    # Output should be equal
    # print("Total Words: ", len(all_words_proc), "\nTotal Fixation points: ", len(all_words))
    
    
    seq_len = config.gaze_seq_len
    
    chunks_words = [all_words_proc[x:x+seq_len] for x in range(0, len(all_words_proc), seq_len)]
    chunks_fixation = [fixation[x:x+seq_len] for x in range(0, len(fixation), seq_len)]
    
    return chunks_words, chunks_fixation

def get_provo():

    chunks_words, chunks_fixation, chunks_pos_tags = preprocessing_provo()
    #equal data points
    # print("Chunked words Points: ", len(chunks_words),
    #       "\nChunked fixations Points: ", len(chunks_fixation),
    #       "\nChunked POS Tags: ", len(chunks_pos_tags))
    
    
    data = pd.DataFrame()  
    data['chunked_words'] = [" ".join(x) for x in chunks_words] 
    data['label'] = chunks_fixation 
    # data['idx'] = np.arange(0,len(chunks_fixation)) 
 
    # Split DataFrame randomly
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print("Provo Dataset")

    print("Train DataFrame:")
    print(train_df.shape)
    train_df = train_df.reset_index()
    
    print("Validation DataFrame:")
    print(val_df.shape)
    val_df = val_df.reset_index()
    
    print("Test DataFrame:")
    print(test_df.shape)
    test_df = test_df.reset_index()

    return train_df, val_df, test_df

            



def get_geco():

    chunks_words, chunks_fixation = preprocessing_geco()
    
    #equal data points
    print("Chunked words Points: ", len(chunks_words),"\nChunked fixations Points: ", len(chunks_fixation))

    
    data = pd.DataFrame()  
    data['chunked_words'] = [" ".join(x) for x in chunks_words] 
    data['label'] = chunks_fixation 
    # data['idx'] = np.arange(0,len(chunks_fixation)) 
 
    # Split DataFrame randomly
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print("GECO Dataset")

    print("Train DataFrame:")
    print(train_df.shape)
    train_df = train_df.reset_index()
    
    print("Validation DataFrame:")
    print(val_df.shape)
    val_df = val_df.reset_index()
    
    print("Test DataFrame:")
    print(test_df.shape)
    test_df = test_df.reset_index()

    return train_df, val_df, test_df

def get_gaze_data_complete():
    
    chunks_words_geco, chunks_fixation_geco = preprocessing_geco()

    chunks_words_provo, chunks_fixation_provo, chunks_pos_tags_provo = preprocessing_provo()

    df_iitb_train, df_iitb_test, df_iitb_val = get_iitb_gaze_hallucination()


    chunks_words = chunks_words_geco + chunks_words_provo #+ chunks_words_iitb
    chunks_fixation = chunks_fixation_geco + chunks_fixation_provo #+ chunks_fixation_iitb

    data = pd.DataFrame()  
    data['chunked_words'] = [" ".join(x) for x in chunks_words] 
    data['label'] = chunks_fixation 
    # data['idx'] = np.arange(0,len(chunks_fixation)) 
 
    # Split DataFrame randomly
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_df = pd.concat([train_df, df_iitb_train]).reset_index(drop=True)
    val_df = pd.concat([val_df, df_iitb_val]).reset_index(drop=True)
    test_df = pd.concat([test_df, df_iitb_test]).reset_index(drop=True)

    
    print("GECO + PROVO + IITB Dataset")

    print("Train DataFrame:")
    print(train_df.shape)
    train_df = train_df.reset_index()
    
    print("Validation DataFrame:")
    print(val_df.shape)
    val_df = val_df.reset_index()
    
    print("Test DataFrame:")
    print(test_df.shape)
    test_df = test_df.reset_index()

    return train_df, val_df, test_df


