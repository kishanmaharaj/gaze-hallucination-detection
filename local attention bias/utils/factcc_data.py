import pandas as pd

def encoding_label(data, column_name, encoding_seq):
    
    """
    data: Pandas DataFrame
    column_name: String (Name of the column for encoding)
    encoding_seq: List (Example: ["CORRECT", "INCORRECT"] will be encoded to [0, 1]) 
    """
    
    encoded_label = [0 if x ==encoding_seq[0] else 1 for x in data['label']]
    data['encoded_label'] = encoded_label
    return data



def get_factcc_data(print_distribution=True, complete_text=False):
    
    # load Dataset Labse
    train_data_ab_labse =  pd.read_csv("data/factcc/train_labse_full.csv")
    # dev_data_ab_labse = pd.read_csv("data/factcc/dev_ann_labse.csv")
    # test_data_ab_labse = pd.read_csv("data/factcc/test_ann_labse.csv")

    #Load dataset Ensemble
    train_data_ab =  pd.read_csv("data/factcc/train_ann_ensemble.csv")
    dev_data_ab = pd.read_csv("data/factcc/dev_ann_ensemble.csv")
    test_data_ab = pd.read_csv("data/factcc/test_ann_ensemble.csv")
    
    train_data_ab_labse = encoding_label(data=train_data_ab_labse, column_name="label", encoding_seq=["CORRECT", "INCORRECT"])
    dev_data_ab = encoding_label(data=dev_data_ab, column_name="label", encoding_seq=["CORRECT", "INCORRECT"])
    test_data_ab = encoding_label(data=test_data_ab, column_name="label", encoding_seq=["CORRECT", "INCORRECT"])
    
    
    if complete_text:

        print("Including complete text")
        
        # Train
        df_train = pd.DataFrame()
        df_train['sentence1'] = train_data_ab_labse['text']
        df_train['sentence2'] = train_data_ab_labse['claim']
        df_train['label'] = train_data_ab_labse['encoded_label']
        df_train['idx'] = train_data_ab_labse.index.tolist()
        
        # Test
        df_test = pd.DataFrame()
        df_test['sentence1'] = test_data_ab['text']
        df_test['sentence2'] = test_data_ab['claim']
        df_test['label'] = test_data_ab['encoded_label']
        df_test['idx'] = test_data_ab.index.tolist()
        
        # Validation
        df_val = pd.DataFrame()
        df_val['sentence1'] = dev_data_ab_labse['text']
        df_val['sentence2'] = dev_data_ab_labse['claim']
        df_val['label'] = dev_data_ab_labse['encoded_label']
        df_val['idx'] = dev_data_ab_labse.index.tolist()

    else:
        
        print("Including Global Attention bias text from ensemble")

        # Train
        df_train = pd.DataFrame()
        df_train['sentence1'] = train_data_ab['most_voted_sent']
        df_train['sentence2'] = train_data_ab['claim']
        df_train['label'] = train_data_ab_labse['encoded_label']
        df_train['idx'] = train_data_ab.index.tolist()
        
        # Test
        df_test = pd.DataFrame()
        df_test['sentence1'] = test_data_ab['final_sent']
        df_test['sentence2'] = test_data_ab['claim']
        df_test['label'] = test_data_ab['encoded_label']
        df_test['idx'] = test_data_ab.index.tolist()
        
        # Validation
        df_val = pd.DataFrame()
        df_val['sentence1'] = dev_data_ab['final_sent']
        df_val['sentence2'] = dev_data_ab['claim']
        df_val['label'] = dev_data_ab['encoded_label']
        df_val['idx'] = dev_data_ab.index.tolist()
    

    
    if print_distribution:
        ## Print Distribution
        print("----"*10)
        print("FACTCC Data Distribution")
        print("----"*10)

        ## Train Distribution
        print("Train dataset (Total):           \t", len(train_data_ab))
        print("Train dataset (Non-hallucinated):\t", sum(train_data_ab_labse["encoded_label"]==0))
        print("Train dataset (Hallucinated):    \t", sum(train_data_ab_labse["encoded_label"]==1))
        print("----"*10)


        ## Test Distribution
        print("Test dataset (Total):           \t", len(test_data_ab))
        print("Test dataset (Non-hallucinated):\t", sum(test_data_ab["encoded_label"]==0))
        print("Test dataset (Hallucinated):    \t", sum(test_data_ab["encoded_label"]==1))
        print("----"*10)

        ## Validation Distribution
        print("Validation dataset (Total):           \t", len(dev_data_ab))
        print("Validation dataset (Non-hallucinated):\t", sum(dev_data_ab["encoded_label"]==0))
        print("Validation dataset (Hallucinated):    \t", sum(dev_data_ab["encoded_label"]==1))

        print("----"*10)
        
        print("Train set: ", df_train.shape)
        print("Validation set: ", df_val.shape)
        print("Test set: ", df_test.shape)
        print("----"*10)

    
    return df_train, df_test, df_val
