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
import keras
#for visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from IPython.display import HTML as html_print
from IPython.display import display
import matplotlib.patheffects as path_effects

#define custom attention layer
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

filename = "checkpoint.model.keras"
model = load_model(filename, custom_objects={'AttLayer': AttLayer})
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


#data 

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

#process data
def get_train_test(data, eval_data, x_text, eval_text, labels, eval_labels):
    
    X_train=x_text
    Y_train=labels

    X_test=eval_text
    Y_test=eval_labels

    # for train data
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
    
    Y_test_encoded = [label_mapping[label] for label in Y_test] 
  
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


    trainY = to_categorical(Y_train_encoded_final, nb_classes=NUM_CLASSES)
    testY = to_categorical(Y_test_encoded_final, nb_classes=NUM_CLASSES)

    testY = testY.astype(np.float64)
    trainY = trainY.astype(np.float64)
    print(f'testy.dtype {testY.dtype}')   # Check data type of testY
    print(f'trainy.dtype {trainY.dtype}')   # Check data type of trainY
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=eval_max_document_length, value=0.)
    
    data_dict = {
        "data": data,
        "trainX" : trainX,
        "trainY" : trainY,
        "testX" : testX,
        "testY" : testY,
        "vocab_processor" : vocab_processor
    }
    return data_dict

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
print(testY[:5])


attention_model = Model(inputs=model.input,
                        outputs=[model.output, model.get_layer('att_layer_1').output])


predictions, attention_weights_list = attention_model.predict(testX_padded)

# Create a color map for coloring text based on attention weights
cmap = plt.cm.YlGnBu

# Create a scalar mappable to map normalized attention weights to colors
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(cmap=cmap, norm=norm)

# Iterate over each text in the list and visualize attention weights
for i, text in enumerate(x_text):
    attention_weights = attention_weights_list[i]
    
    # Normalize attention weights to range [0, 1]
    normalized_weights = (attention_weights - attention_weights.min()) / (attention_weights.max() - attention_weights.min())
    
    # Plot the text with colored words
    plt.figure(figsize=(10, 2))
    ax = plt.gca()
    ax.set_xlim(0, len(text.split()) - 1)  # Adjust x-axis limit based on the number of words
    ax.set_ylim(0, 1)  # Limit y-axis to 1 to ensure the text is centered vertically

    # Iterate over each word in the text
    for j, (word, weight) in enumerate(zip(text.split(), normalized_weights)):
        # Get the color corresponding to the attention weight
        color = sm.to_rgba(weight)
        
        # Calculate position for the current word
        x = j
        y = 0.5  # Center vertically
        
        # Create a Text object with automatic wrapping
        text_obj = ax.text(x, y, word, color=color, fontsize=12, ha='center', va='center', bbox=dict(facecolor='none', edgecolor='none', pad=0))
        text_obj.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'), path_effects.Normal()])  # Add stroke effect for better visibility
        
    # Set plot parameters
    # ax.set_xticks(np.arange(len(text.split())))
    # ax.set_xticklabels(text.split(), rotation=45, ha='right', va='top')  # Rotate and align x-axis labels
    ax.set_yticks([])
    plt.title('Attention Visualization for Text {}'.format(i+1))
    
    # Add color bar for legend
    plt.colorbar(sm, orientation='horizontal', label='Attention Weight')
    
    plt.show()