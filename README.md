**all_code.py** Contain main code for training models

**model.py**    Defines functions needed to create models

**evaluate_saved_model.py**  Contains code to load saved model and test on new data

**data folder**  Should be where you required the data files are saved in pkl format for loading

**checkpoint.model.keras** saved model keras file

**weights_visualise2.py** contains code to visualise attenton weights


To load data in any of the scripts you need to copy paste the train and test sets you want with the names formspring_data.pkl and eval_formspring_data.pkl into the data folder. 

The directory of word_vectors is empty because the glove embeddings could not be uploaded due to their size, but they can be downloaded from [here](https://nlp.stanford.edu/data/glove.840B.300d.zip). Rename the unzipped 300d file to glove.6B.300d.txt. 

The original code by can be accessed in [GitHub](https://github.com/sweta20/Detecting-Cyberbullying-Across-SMPs). 


**Library Requirements:** 
Python 3.7
keras==2.3.1
tensorflow==1.13.1 
tflearn==0.3.2
scikit-learn==0.21.3
pandas==0.25.2
tqdm==4.36.1
numpy=1.21.6   
