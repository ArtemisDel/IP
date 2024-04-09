# imports. 
from models import get_model
import argparse
import pickle
import string
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
import preprocessor as p
from collections import Counter
import os
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from tensorflow.contrib import learn
from tflearn.data_utils import to_categorical, pad_sequences
from scipy import stats
import tflearn
import json
# my stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import pdb
import tensorflow as tf


# block 1. 
def get_filename(dataset):
    global NUM_CLASSES, HASH_REMOVE
    if(dataset=="twitter"):
        NUM_CLASSES = 3 #! CHANGE HERE TO 2 from 3
        HASH_REMOVE = True
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
        if(HASH_REMOVE):
            x_text.append(p.tokenize((data[i]['text']).encode('utf-8')))
        else:
        #     x_text.append(data[i]['body']) #change here from 'text' to 'body'
        # labels.append(data[i]['level_1']) #changed here from 'label' to 'level_1'
            x_text.append(data.loc[i]['body'])
        labels.append(data.loc[i]['level_1'])
        

    return x_text,labels


def get_eval_filename(dataset):
    global NUM_CLASSES, HASH_REMOVE
    if(dataset=="twitter"):
        NUM_CLASSES = 3 #! CHANGE HERE TO 2 from 3
        HASH_REMOVE = True
        eval_filename = "data/twitter_data.pkl"
    elif(dataset=="eval_formspring"):
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
        if(HASH_REMOVE):
            val_text.append(p.tokenize((data[i]['text']).encode('utf-8')))
        else:
        #     x_text.append(data[i]['body']) #change here from 'text' to 'body'
        # labels.append(data[i]['level_1']) #changed here from 'label' to 'level_1'
            val_text.append(data.loc[i]['body'])
        val_labels.append(data.loc[i]['level_1'])
    return val_text, val_labels    



# block 2. 
def get_embedding_weights(filename, sep):
	embed_dict = {}
	file = open(filename,'r')
	for line in file.readlines():
	    row = line.strip().split(sep)
	    embed_dict[row[0]] = row[1:]
	print('Loaded from file: ' + str(filename))
	file.close()
	return embed_dict

def map_embedding_weights(embed, vocab, embed_size):
    vocab_size = len(vocab)
    # pdb.set_trace()
    embeddingWeights = np.zeros((vocab_size , embed_size))
    # pdb.set_trace()
    n = 0
    words_missed = []
    for k, v in vocab.items(): #change iteritems to items
        # pdb.set_trace()
        try:
            embeddingWeights[v] = embed[k]
            # pdb.set_trace()
        except:
            n += 1
            words_missed.append(k)
            pass
            # pdb.set_trace()
    print("%d embedding missed"%n, " of " , vocab_size)
    return embeddingWeights

def get_embeddings_dict(vector_type, emb_dim):
    if vector_type == 'sswe':
        emb_dim==50
        sep = '\t'
        vector_file = 'word_vectors/sswe-u.txt'
    elif vector_type =="glove":
        sep = ' '
        if data == "wiki" or data == 'formspring':
            vector_file = 'word_vectors/glove.6B.' + str(emb_dim) + 'd.txt'
        else:
            vector_file = 'word_vectors/glove.twitter.27B.' + str(emb_dim) + 'd.txt'
    else:
        print("ERROR: Please specify a correst model or SSWE cannot be loaded with embed size of: " + str(emb_dim) )
        return None
    
    embed = get_embedding_weights(vector_file, sep)
    return embed


#block 3
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

# block 4. 
def dump_learned_embedding(data, model_type, vector_type, embed_size, embed, vocab_processor):
    vocab = vocab_processor.vocabulary_._mapping
    vocab_size = len(vocab)
    embedDict = {}
    n = 0
    words_missed = []
    for k, v in vocab.items(): #change iteritems to items
        # pdb.set_trace()
        try:
            embeddingDict[v] = embed[k]
            # pdb.set_trace()
        except:
            n += 1
            words_missed.append(k)
            # pdb.set_trace()
            pass
    # pdb.set_trace()
    print("%d embedding missed"%n, " of " , vocab_size)
    
    filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + embed_size + ".pkl"
    with open(filename, 'wb') as handle:
        pickle.dump(embedDict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# block 6. 
def return_data(data_dict):
    return data_dict["data"], data_dict["trainX"], data_dict["trainY"], data_dict["testX"], data_dict["testY"], data_dict["vocab_processor"]

# block 7.a 
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)

# block 7.b 
def train(data_dict, model_type, vector_type, embed_size, dump_embeddings=False):

    #NEW stuff starts here
    checkpoint_filepath = '/disk/scratch/s1955786/IP2/Model_code2/checkpoint.model.keras' #changed filepath to end with weights.h5 as in https://keras.io/api/callbacks/model_checkpoint/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False, #set this to true to only save weights
    monitor='val_accuracy', #changed here from val_acc to val_accuracy apparently they are different
    mode='max',
    save_best_only=False) 

    #NEW stuff ends here

    data, trainX, trainY, testX, testY, vocab_processor = return_data(data_dict)

    
    vocab_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocab_size))
    vocab = vocab_processor.vocabulary_._mapping
  
    
    print("Running Model: " + model_type + " with word vector initiliazed with " + vector_type + " word vectors.")
    model = get_model(model_type, trainX.shape[1], vocab_size, embed_size, NUM_CLASSES, LEARN_RATE)

    initial_weights = model.get_weights()
    shuffle_weights(model, initial_weights)
    # pdb.set_trace()

    if(model_type == 'cnn'):
        if(vector_type!="random"):
            print("Word vectors used: " + vector_type)
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
            model.set_weights(embeddingWeights, map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size))
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
        else:
            model.fit(trainX, trainY, n_epoch = EPOCHS, shuffle=True, show_metric=True, batch_size=BATCH_SIZE)
    else:
        if(vector_type!="random"):
            print("Word vectors used: " + vector_type)
            model.layers[0].set_weights([map_embedding_weights(get_embeddings_dict(vector_type, embed_size), vocab, embed_size)])
            model.fit(trainX, trainY, epochs=EPOCHS, callbacks=[model_checkpoint_callback], shuffle=True, batch_size=BATCH_SIZE, 
                  verbose=1)
        else:
            model.fit(trainX, trainY, epochs=EPOCHS, callbacks=[model_checkpoint_callback], shuffle=True, batch_size=BATCH_SIZE, 
                  verbose=1)
            
    if (dump_embeddings==True):
        if(model_type == 'cnn'):
            embeddingWeights = tflearn.get_layer_variables_by_name('EmbeddingLayer')[0]
        else:
            embed = model.layers[0].get_weights()[0]
    
        embed_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + ".pkl"
        embed.dump(embed_filename)
        
        vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_dict.json"
        reverse_vocab_filename = output_folder_name + data + "_" + model_type + "_" + vector_type + "_" + str(embed_size) + "_reversedict.json"
        
        with open(vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._mapping, fp)
        with open(reverse_vocab_filename, 'w') as fp:
            json.dump(vocab_processor.vocabulary_._reverse_mapping, fp)
    
    return  evaluate_model(model, testX, testY)

# block 8. 
def print_scores(precision_scores, recall_scores, f1_scores):
    for i in range(NUM_CLASSES):
        print("\nPrecision Class %d (avg): %0.3f (+/- %0.3f)" % (i, precision_scores[:, i].mean(), precision_scores[:, i].std() * 2))
        print( "\nRecall Class %d (avg): %0.3f (+/- %0.3f)" % (i, recall_scores[:, i].mean(), recall_scores[:, i].std() * 2))
        print( "\nF1 score Class %d (avg): %0.3f (+/- %0.3f)" % (i, f1_scores[:, i].mean(), f1_scores[:, i].std() * 2))

# block 9. 

# def oversample():


def get_data(data, oversampling_rate):
    
    x_text, labels = load_data(get_filename(data)) 
 
    if(data=="twitter"):
        NUM_CLASSES = 3
        dict1 = {'racism':2,'sexism':1,'none':0}
        labels = [dict1[b] for b in labels]
        
        racism = [i for i in range(len(labels)) if labels[i]==2]
        sexism = [i for i in range(len(labels)) if labels[i]==1]
        x_text = x_text + [x_text[x] for x in racism]*(oversampling_rate-1)+ [x_text[x] for x in sexism]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
    
    else:  
        NUM_CLASSES = 2
        # dict1 = {'Misogynistic':1,'Nonmisogynistic':0}
        # labels = [dict1[b] for b in labels]
        # pdb.set_trace()
        bully = [i for i in range(len(labels)) if labels[i]=="Misogynistic"] #label 1 corresponds to misogynistic label
        print("Counter before oversampling")
        # pdb.set_trace()
        from collections import Counter
        print(Counter(labels))
        for i in bully:
            for _ in range(oversampling_rate - 1):
                x_text.append(x_text[i])
                labels.append("Misogynistic")
                # pdb.set_trace()
        print("Counter after oversampling")
        from collections import Counter
        print(Counter(labels))
        # pdb.set_trace()

        filter_data = []
        for text in x_text:
            filter_data.append("".join(str(l) for l in text if l not in string.punctuation))

        # for text, label in zip(x_text, labels):
        #     print("Text:", text)
        #     print("Label:", label)
        # # pdb.set_trace()  
        # pdb.set_trace()      
    return x_text, labels

# block 9. 
def get_eval_data(eval_data):
    # pdb.set_trace()
    val_text, val_labels = load_eval_data(get_eval_filename(eval_data)) 
    # pdb.set_trace()
 
    if(data=="twitter"):
        NUM_CLASSES = 3
        dict1 = {'racism':2,'sexism':1,'none':0}
        labels = [dict1[b] for b in labels]
        
        racism = [i for i in range(len(labels)) if labels[i]==2]
        sexism = [i for i in range(len(labels)) if labels[i]==1]
        val_text = x_text + [val_text[x] for x in racism]*(oversampling_rate-1)+ [val_text[x] for x in sexism]*(oversampling_rate-1)
        labels = labels + [2 for i in range(len(racism))]*(oversampling_rate-1) + [1 for i in range(len(sexism))]*(oversampling_rate-1)
    
    else:  
        NUM_CLASSES = 2
        # bully = [i for i in range(len(labels)) if labels[i]==1]
        val_text = val_text #+ [x_text[x] for x in bully]*(oversampling_rate-1)
        val_labels = list(val_labels) #+ [1 for i in range(len(bully))]*(oversampling_rate-1)

    print("Counter after oversampling evaluation data")
    from collections import Counter
    print(Counter(val_labels))
    
    filter_valdata = []
    for text in val_text:
        filter_valdata.append("".join(str(l) for l in text if l not in string.punctuation))
        
        
    return val_text, val_labels

# block 10. 
models = [ 'cnn', 'lstm', 'blstm', 'blstm_attention']
word_vectors = ["random", "glove" ,"sswe"]#!I'm leaning on glove embeddings just because im not sure sentiment-secific embeddings is the way to go
EPOCHS = 10
BATCH_SIZE = 128
MAX_FEATURES = 2
NUM_CLASSES = None
DROPOUT = 0.25 #? change here to finetune
LEARN_RATE = 0.01 #change here to finetune!!!
HASH_REMOVE = None
output_folder_name = "results/"

# block 10a. 
def get_train_test(data, eval_data, x_text, eval_text, labels, eval_labels):
    
    
    # X_train, Y_train, Y_test = train_test_split( x_text, labels, random_state=42, test_size=0.10)
    X_train=x_text
    Y_train=labels

    X_test=eval_text
    Y_test=eval_labels

    #for train data
    post_length = np.array([len(x.split(" ")) for x in x_text])
    if(data != "twitter"):
        max_document_length = int(np.percentile(post_length, 95))
    else:
        max_document_length = max(post_length)
    print("Document length : " + str(max_document_length))

    #for evaluation data
    eval_post_length = np.array([len(x.split(" ")) for x in eval_text])
    if(data != "twitter"):
        eval_max_document_length = int(np.percentile(post_length, 95))
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
    Y_train_encoded = [label_mapping[label] for label in Y_train]
    Y_test_encoded = [label_mapping[label] for label in Y_test]  
    # pdb.set_trace()  
    Y_train_encoded_final = label_encoder.fit_transform(Y_train_encoded)
    Y_test_encoded_final = label_encoder.transform(Y_test_encoded)
    # print("Original Y_train labels:", Y_train)
    # print("Encoded Y_train labels:", Y_train_encoded_final)

    print("Original Y_test labels:", Y_test)
    print("Encoded Y_test labels:", Y_test_encoded_final)
    for label, encoded_label in label_mapping.items():
        print(f"Label: {label}, Encoded value: {encoded_label}")



    #pdb.set_trace()


    # Convert labels to one-hot encoded vectors
    #trainY = to_categorical(Y_train_encoded, num_classes=NUM_CLASSES)
    #testY = to_categorical(Y_test_encoded, num_classes=NUM_CLASSES)
    trainY = to_categorical(Y_train_encoded_final, nb_classes=NUM_CLASSES)
    testY = to_categorical(Y_test_encoded_final, nb_classes=NUM_CLASSES)

    # trainY = np.asarray(Y_train)
    # testY = np.asarray(Y_test)
    
    print(trainY.dtype)  # Check data type of trainY
    print(testY.dtype)   # Check data type of testY
        
    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=eval_max_document_length, value=0.)
    #pdb.set_trace()
    #trainY = to_categorical(trainY, nb_classes=NUM_CLASSES)
    #testY = to_categorical(testY, nb_classes=NUM_CLASSES)
    
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

# block 11. 
def run_model(data, eval_data, oversampling_rate, model_type, vector_type, embed_size):    
    x_text, labels = get_data(data, oversampling_rate)
    eval_text, eval_labels=get_eval_data(eval_data)
    data_dict = get_train_test(data, eval_data, x_text, eval_text,labels, eval_labels)
    precision, recall, f1_score = train(data_dict, model_type, vector_type, embed_size)
    
    #!NEED HERE TO ADD CODE TO SAVE MODEL CHECKPOINTS
    #? check out "https://keras.io/api/models/model_saving_apis/"

# run model. 
data = "formspring"
eval_data="eval_formspring"
model_type = "blstm_attention"
vector_type = "glove"

# for embed_size in [25, 50, 100, 200]: 
for embed_size in [300]: 
    run_model(data, eval_data, 9, model_type, vector_type, embed_size)
    #set oversampling rate here as 9 because 4560/538=8.45 ~9




