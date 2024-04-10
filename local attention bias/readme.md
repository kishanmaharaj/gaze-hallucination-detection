## Local Attention Bias
The "models" folder can be downloaded from the link below: 
```
https://drive.google.com/file/d/1nCBlU6JoH_rHWqKa04-3Aq4vq3yOS2ZR/view?usp=sharing
```
Please note that the downloaded file will be a compressed folder (.zip file) containing the "models" directory. 
After decompressing the .zip file, please keep the obtained "models" folder in this directory. 

The content of the "models" directory is described here: 
```
models
│   bert-base-uncased_tsm_False_fix_False.pt # GAB 
│   bert-base-uncased_tsm_True_fix_False.pt  # GAB+LAB
│   bert-base-uncased_tsm_True_fix_True.pt   # GAB+LAB+Gaze
│   
└───gaze_model
    │   bert-base-uncased_gaze.pt # Gaze Model trained on fixation data.
```


We will soon shift all our models to huggingface. 
