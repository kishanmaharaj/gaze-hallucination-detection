import config
import numpy as np 
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_geco():
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
    print("Total Words: ", len(all_words_proc), "\nTotal Fixation points: ", len(all_words))
    
    
    seq_len = config.gaze_seq_len
    
    chunks_words = [all_words_proc[x:x+seq_len] for x in range(0, len(all_words_proc), seq_len)]
    chunks_fixation = [fixation[x:x+seq_len] for x in range(0, len(fixation), seq_len)]
    
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