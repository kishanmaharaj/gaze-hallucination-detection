import config
import numpy as np 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_provo():
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