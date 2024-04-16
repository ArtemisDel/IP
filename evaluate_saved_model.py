import tflearn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
import tensorflow as tf
import os
os.environ['KERAS_BACKEND']='theano'
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout, LSTM, GRU, Bidirectional
# here changed Merge to merge because it wouldn't import
from keras.models import Model,Sequential
from keras.engine.topology import Layer, InputSpec
from keras import initializers, optimizers
from keras.models import load_model
from keras import backend as K
import re
import pdb
import pickle
from tensorflow.contrib import learn
from sklearn.preprocessing import LabelEncoder
from tflearn.data_utils import to_categorical, pad_sequences
import theano
# theano.config.mode = 'DEBUG_MODE'
# theano.config.exception_verbosity = 'high'
import keras
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score
import pandas as pd

print(f"Imports loaded successfully")

#define data
eval_data="eval_formspring"
data="formspring"
MAX_FEATURES = 2

def get_filename(dataset):
    global NUM_CLASSES
    if(dataset=="twitter"):
        NUM_CLASSES = 3 #! CHANGE HERE TO 2 from 3
        # HASH_REMOVE = True
        filename = "data/twitter_data.pkl"
    elif(dataset=="formspring"):
        NUM_CLASSES = 2
        filename = "data/formspring_data.pkl"
    elif(dataset=="wiki"):
        NUM_CLASSES = 2
        filename = "data/wiki_data.pkl"
    return filename


def load_data(filename):
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    x_text = []
    labels = [] 
    for i in range(len(data)):
        # if(HASH_REMOVE):
        #     x_text.append(p.tokenize((data[i]['text']).encode('utf-8')))
        # else:
        # #     x_text.append(data[i]['body']) #change here from 'text' to 'body'
        # labels.append(data[i]['level_1']) #changed here from 'label' to 'level_1'
        x_text.append(data.loc[i]['body'])
        labels.append(data.loc[i]['level_1'])
        

    return x_text,labels

def get_eval_filename(dataset):
    global NUM_CLASSES
    if(dataset=="eval_formspring"):
        NUM_CLASSES = 2
        # pdb.set_trace()
        eval_filename = "data/eval_formspring_data.pkl"
        # pdb.set_trace()
    elif(dataset=="wiki"):
        NUM_CLASSES = 2
        eval_filename = "data/wiki_data.pkl"
    return eval_filename

def load_eval_data(filename):
    print("Loading data from file: " + filename)
    data = pickle.load(open(filename, 'rb'))
    val_text = []
    val_labels = [] 
    for i in range(len(data)):
        #     x_text.append(data[i]['body']) #change here from 'text' to 'body'
        # labels.append(data[i]['level_1']) #changed here from 'label' to 'level_1'
        val_text.append(data.loc[i]['body'])
        val_labels.append(data.loc[i]['level_1'])
    return val_text, val_labels    

class AttLayer(Layer):

    def __init__(self, **kwargs):
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],),
                                      initializer='random_normal',
                                      trainable=True)
        super(AttLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# load model from file. 
new_model=load_model("checkpoint.model.keras", custom_objects={'AttLayer': AttLayer})
# pdb.set_trace()
print(new_model.summary())
pdb.set_trace()

# get eval data
# def get_eval_data(eval_data):
#     pdb.set_trace()
#     eval_text, eval_labels = load_eval_data(get_eval_filename(eval_data)) 
#     pdb.set_trace()
 
#     NUM_CLASSES = 2
#     # bully = [i for i in range(len(labels)) if labels[i]==1]
#     eval_text = eval_text #+ [x_text[x] for x in bully]*(oversampling_rate-1)
#     eval_labels = list(eval_labels) #+ [1 for i in range(len(bully))]*(oversampling_rate-1)

#     print("Counter after oversampling evaluation data")
#     from collections import Counter
#     print(Counter(eval_labels))
    
#     # filter_valdata = []
#     # for text in val_text:
#     #     filter_valdata.append("".join(str(l) for l in text if l not in string.punctuation))
        
        
#     return eval_text, eval_labels


#process eval data
def get_train_test(data, eval_data, x_text, eval_text, labels, eval_labels):
    
    
    # X_train, Y_train, Y_test = train_test_split( x_text, labels, random_state=42, test_size=0.10)
    X_train=x_text
    Y_train=labels

    X_test=eval_text
    Y_test=eval_labels

    # #for train data
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    #for evaluation data
    eval_post_length = np.array([len(x.split(" ")) for x in eval_text])
    if(eval_data != "twitter"):
        eval_max_document_length = int(np.percentile(eval_post_length, 95))
    else:
        eval_max_document_length = max(eval_post_length)
    print("Document length : " + str(eval_max_document_length))
    
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, MAX_FEATURES)
    vocab_processor = vocab_processor.fit(x_text)

    trainX = np.array(list(vocab_processor.transform(X_train)))
    testX = np.array(list(vocab_processor.transform(X_test)))
    
    # Perform label encoding
    label_encoder = LabelEncoder()
    label_mapping={"Misogynistic":1, "Nonmisogynistic":0} #label mapping
    label_mapping = {label: int(encoded_label) for label, encoded_label in label_mapping.items()}
    for label, encoded_label in label_mapping.items():
        print(f"Type of {label}: {type(encoded_label)}")
    Y_train_encoded = [label_mapping[label] for label in Y_train]
    # label_encoder.fit(Y_test)
    Y_test_encoded = [label_mapping[label] for label in Y_test] 
    # pdb.set_trace()  
    # Y_train_encoded_final = label_encoder.fit_transform(Y_train_encoded)
    # Y_test_encoded_final = label_encoder.transform(Y_test_encoded)
    Y_test_encoded_final=Y_test_encoded
    Y_train_encoded_final = Y_train_encoded
    print(type(Y_test_encoded_final))
    print(type(Y_train_encoded_final))
    print("Original Y_train labels:", Y_train)
    print("Encoded Y_train labels:", Y_train_encoded_final)

    print("Original Y_test labels:", Y_test)
    print("Encoded Y_test labels:", Y_test_encoded_final)
    for label, encoded_label in label_mapping.items():
        print(f"Label: {label}, Encoded value: {encoded_label}")



    #pdb.set_trace()


    # Convert labels to one-hot encoded vectors
    # trainY = to_categorical(Y_train_encoded, num_classes=NUM_CLASSES)
    # testY = to_categorical(Y_test_encoded, num_classes=NUM_CLASSES)
    trainY = to_categorical(Y_train_encoded_final, nb_classes=NUM_CLASSES)
    testY = to_categorical(Y_test_encoded_final, nb_classes=NUM_CLASSES)

    # trainY = np.asarray(Y_train)
    # testY = np.asarray(Y_test)
    
    # print(trainY.dtype)  # Check data type of trainY
    testY = testY.astype(np.float64)
    trainY = trainY.astype(np.float64)
    print(f'testy.dtype {testY.dtype}')   # Check data type of testY
    print(f'trainy.dtype {trainY.dtype}')   # Check data type of trainY
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=eval_max_document_length, value=0.)
    #pdb.set_trace()
    # trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    # testY = to_categorical(testY, nb_classes=NUM_CLASSES)
    
    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    # pdb.set_trace()
    return data_dict

def evaluate_model(model, testX, testY):
    temp = model.predict(testX)
    # what values exist in in temp? 
    # pdb.set_trace()
    y_pred  = np.argmax(temp, 1)
    y_true = np.argmax(testY, 1)
    precision = metrics.precision_score(y_true, y_pred, average=None)
    recall = metrics.recall_score(y_true, y_pred, average=None)
    f1_score = metrics.f1_score(y_true, y_pred, average=None)
    print("Precision: " + str(precision) + "\n")
    print("Recall: " + str(recall) + "\n")
    print("f1_score: " + str(f1_score) + "\n")
    print(confusion_matrix(y_true, y_pred))
    print(":: Classification Report")
    print(classification_report(y_true, y_pred))
    return precision, recall, f1_score

def return_data(data_dict):
    return data_dict["data"],data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"], data_dict["vocab_processor"]

x_text, labels = load_data(get_filename(data)) 
# pdb.set_trace()
eval_text, eval_labels = load_eval_data(get_eval_filename(eval_data)) 
# pdb.set_trace()
data_dict=get_train_test(data, eval_data, x_text, eval_text, labels, eval_labels)
# pdb.set_trace()

data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)

#dynamically pad sequences
max_sequence_length = 303 #from the model
testX_padded = pad_sequences(testX, maxlen=max_sequence_length, padding='post')
trainX_padded= pad_sequences(trainX, maxlen=max_sequence_length, padding='post')
print(testX_padded.shape)
print(testY.shape)
print(trainX_padded.shape)
print(trainY.shape)
print("Input Data (testX_padded):")
print(testX_padded[:5])  # Print the first 5 samples

print("Labels (testY):")
print(testY[:5])  # Print the first 5 labels

for layer in new_model.layers:
    print(layer.get_config())

pdb.set_trace()
#compile model to ready for binary classification
learn_rate = 0.01
adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999)
pdb.set_trace()
#define precision and recall
# new_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# temp = new_model.predict(trainX_padded)
# y_pred  = np.argmax(temp, 1)
# y_true = np.argmax(trainY, 1)
# precision = metrics.precision_score(y_true, y_pred, average=None)
# recall = metrics.recall_score(y_true, y_pred, average=None)
# f1_score = metrics.f1_score(y_true, y_pred, average=None)
#recompile model + custom metrics
# define precision and recall 
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    f1_score = 2 * ((precision_value * recall_value) / (precision_value + recall_value + K.epsilon()))
    return f1_score
new_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy',precision, recall, f1_score])
pdb.set_trace()
keras.utils.plot_model(new_model, show_shapes=True)
pdb.set_trace()
# evaluate_model(new_model, testX_padded, testY)
# try:
#     evaluate_model(new_model, trainX_padded, trainY)
# except:
#     print(f"evaluate_model failed")
# pdb.set_trace()
# new_model.evaluate(testX_padded, testY, batch_size=8)
evaluation_results = new_model.evaluate(testX_padded, testY, batch_size=8)

print(new_model.metrics_names)
print(evaluation_results)
pdb.set_trace()

# for i, metric_name in enumerate(new_model.metrics_names[0:]):
#     print(metric_name + ":", evaluation_results[i+1])
# print("test loss, test acc:", results)
# evaluate_model(new_model, testX, testY)   

#get predictions
predictions = new_model.predict(testX_padded)
pdb.set_trace()

predicted_labels = np.argmax(predictions, axis=1)
binary_testY = np.argmax(testY, axis=1)


pdb.set_trace()
# Calculate precision, recall, and F1 score
precision = precision_score(binary_testY, predicted_labels, average=None)
recall = recall_score(binary_testY, predicted_labels, average= None)
f1 = metrics.f1_score(binary_testY, predicted_labels, average=None)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print(classification_report(binary_testY, predicted_labels))

#list to deposit mistakes
incorrect_indices = []
#iterate over predictions to find mismatches between prediction and true category
for i in range(len(predictions)):
    predicted_class = np.argmax(predictions[i])
    true_class = np.argmax(testY[i])
    if predicted_class != true_class:
        incorrect_indices.append(i)

# pdb.set_trace()

#put incorrect predictions  into a dictionary     

incorrect_predictions=[]

for idx in incorrect_indices:
    info = {
        "Index": idx,
        "Predicted_Class": np.argmax(predictions[idx]),
        "True_Class": np.argmax(testY[idx]),
        # Add more information if needed
        # "Sample": x_test[idx],
    }

    incorrect_predictions.append(info)

# pdb.set_trace()    

#convert to dataframe and save as csv
incorrect_df = pd.DataFrame(incorrect_predictions)
incorrect_df.to_csv("incorrect_predictions.csv", index=False)


# run evaluate_model(). 
# within evaluate_model(), note that model results are stored as temp... (1) check format of temp (likely, it's rows as test elements, and columns as true and predicted states), (2) either save as dataframe/csv . 

#print these into a file:
#temp = model.predict(testX)
#y_pred  = np.argmax(temp, 1)
#y_true = np.argmax(testY, 1)
#columns. needed textX, temp, testY
#do a word frequency test on the correctly classified ones








