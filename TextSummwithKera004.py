#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras_self_attention import SeqSelfAttention


# In[2]:


import numpy as np
import pandas as pd 
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


# In[3]:


data=pd.read_csv("C:/Users/Ruba/Downloads/Reviews.csv",nrows=100000)


# In[4]:


data.drop_duplicates(subset=['Text'],inplace=True)#dropping duplicates
data.dropna(axis=0,inplace=True)#dropping na


# In[5]:


data.info()


# In[6]:


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}


# In[7]:


import nltk
nltk.download('stopwords')


# In[8]:


stop_words = set(stopwords.words('english'))


# In[9]:



def text_cleaner(text,num):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    newString = re.sub('[m]{2,}', 'mm', newString)
    if(num==0):
        tokens = [w for w in newString.split() if not w in stop_words]
    else:
        tokens=newString.split()
    long_words=[]
    for i in tokens:
        if len(i)>1:                                                 #removing short word
            long_words.append(i)   
    return (" ".join(long_words)).strip()


# In[10]:


#call the function
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t,0))


# In[11]:


cleaned_text[:5]


# In[12]:


#call the function
cleaned_abstract = []
for t in data['abstract']:
    cleaned_abstract.append(text_cleaner(t,1))


# In[13]:


cleaned_abstract[:10]


# In[14]:


#restore values of text / abstract after cleaning process 
data['cleaned_text']=cleaned_text
data['cleaned_abstract']=cleaned_abstract


# In[15]:


#remove empty spaces and NA ones
data.replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)


# In[16]:



# find the length of sentences in each TEXT / Summay to know the MIN.MAX ranges length
import matplotlib.pyplot as plt

text_word_count = []
abstract_word_count = []

# populate the lists with sentence lengths
for i in data['cleaned_text']:
      text_word_count.append(len(i.split()))

for i in data['cleaned_abstract']:
      abstract_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text':text_word_count, 'abstract':abstract_word_count})

length_df.hist(bins = 30)
plt.show()


# In[17]:


# limit the length of MAX 
max_text_len=30
max_abstract_len=8


# In[18]:


# remove the shorter ones than the MAX (max_text_len) limite above
cleaned_text =np.array(data['cleaned_text'])
cleaned_abstract=np.array(data['cleaned_abstract'])

short_text=[]
short_abstract=[]

for i in range(len(cleaned_text)):
    if(len(cleaned_abstract[i].split())<=max_abstract_len and len(cleaned_text[i].split())<=max_text_len):
        short_text.append(cleaned_text[i])
        short_abstract.append(cleaned_abstract[i])
        
df=pd.DataFrame({'text':short_text,'abstract':short_abstract})


# In[19]:


# add START and END special tokens (StartTok EndToken)
df['abstract'] = df['abstract'].apply(lambda x : 'sostok '+ x + ' eostok')


# In[20]:


# splitting the data 90/10 using train_test_split function
from sklearn.model_selection import train_test_split
x_train,x_validate,y_train,y_validate=train_test_split(np.array(df['text']),np.array(df['abstract']),test_size=0.1,random_state=0,shuffle=True)


# In[21]:


#Text Tokenizer 

from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences

#prepare a tokenizer for reviews on x_train data
x_tokenizer = Tokenizer() 
x_tokenizer.fit_on_texts(list(x_train))


# In[22]:



# find the least used words by count limit here is 4times
threshold=4 

count=0
tot_count=0
freq=0
tot_freq=0

for key,value in x_tokenizer.word_counts.items():
    tot_count=tot_count+1
    tot_freq=tot_freq+value
    if(value<threshold):
        count=count+1
        freq=freq+value
 


# In[23]:


x_tokenizer = Tokenizer(num_words=tot_count-count) 


# In[24]:


print(tot_count-count)


# In[25]:


#prepare a tokenizer for reviews on training data
words_count=tot_count-count #6716
x_tokenizer = Tokenizer(num_words=6716) 
x_tokenizer.fit_on_texts(list(x_train))

#convert text sequences into integer sequences texts_to_sequences
x_train_seq    =   x_tokenizer.texts_to_sequences(x_train) 
x_validate_seq   =   x_tokenizer.texts_to_sequences(x_validate)

#post padding  zero upto maximum length to unify the sentences lengths   
x_train    =   pad_sequences(x_train_seq,  maxlen=max_text_len, padding='post')
x_validate   =   pad_sequences(x_validate_seq, maxlen=max_text_len, padding='post')

#size of vocabulary ( +1 for padding token)
x_vocabulary   =  x_tokenizer.num_words + 1


# In[26]:


x_vocabulary


# In[27]:


#prepare a tokenizer for reviews on training data
y_tokenizer = Tokenizer()   
y_tokenizer.fit_on_texts(list(y_train))


# In[28]:


threshold=6

count=0
tot_count=0
freq=0
tot_freq=0

for key,value in y_tokenizer.word_counts.items():
    tot_count=tot_count+1
    tot_freq=tot_freq+value
    if(value<threshold):
        count=count+1
        freq=freq+value
     


# In[29]:


words_count=tot_count-count
print(words_count)


# In[30]:


#prepare a tokenizer for reviews on training data

words_count=tot_count-count
y_tokenizer = Tokenizer(num_words=words_count) 
y_tokenizer.fit_on_texts(list(y_train))

#convert text sequences into integer sequences using texts_to_sequences
y_train_seq    =   y_tokenizer.texts_to_sequences(y_train) 
y_validate_seq   =   y_tokenizer.texts_to_sequences(y_validate) 

#padding zero upto maximum length
y_train    =   pad_sequences(y_train_seq, maxlen=max_abstract_len, padding='post')
y_validate   =   pad_sequences(y_validate_seq, maxlen=max_abstract_len, padding='post')

#size of vocabulary
y_vocabulary  =   y_tokenizer.num_words +1


# In[31]:


y_tokenizer.word_counts['sostok'],len(y_train)


# In[32]:


from keras import backend as K
from tensorflow.python.framework import ops
ops.reset_default_graph()


# In[33]:


import tensorflow as tf
#K.clear_session()
tf.keras.backend.clear_session()


# In[34]:


from tensorflow.keras.layers import Attention
from attention import AttentionLayer


# In[35]:


from attention import AttentionLayer


# In[ ]:





# In[36]:


from attention import AttentionLayer
import tensorflow as tf
#K.clear_session()
tf.keras.backend.clear_session()
#

latent_dim = 300
embedding_dim=100

# Encoder
encoder_inputs = Input(shape=(30,))

#embedding layer
enc_Embedding =  Embedding(x_vocabulary, embedding_dim,trainable=True)(encoder_inputs)

#encoder lstm 1 
#Return Sequences = True: LSTM produces the hidden state and cell state for every timestep
#Return State = True:LSTM produces the hidden state and cell state of the last timestep only

encoder_lstm01 = LSTM(latent_dim,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
encoder_output01, state_h01, state_c01 = encoder_lstm01(enc_Embedding)

#encoder_output01 will be used soon as input for next layer

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#embedding layer /look up from the vocabulary
dec_emb_layer = Embedding(y_vocabulary, embedding_dim,trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)
#LSTM layer
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,dropout=0.4,recurrent_dropout=0.2)
#Decoder LSTM
decoder_outputs,_, decoder_back_state = decoder_lstm(dec_emb,initial_state=[state_h01, state_c01])

# add Attention layer
 
# then I have to Concat attention input and decoder LSTM output
 

#dense layer
decoder_dense =  TimeDistributed(Dense(y_vocabulary, activation='softmax'))
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model with encoder and decoder 
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()


# In[37]:


tf.keras.layers.Attention


# In[38]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


# In[39]:


#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
es = EarlyStopping(patience=100, monitor='val_loss', restore_best_weights=True) #accuracy, loss, val_loss, val_accuracy


# In[40]:


from tensorflow.keras.callbacks import TensorBoard
tensorboard_callback = TensorBoard(log_dir="./logs", write_graph=True, histogram_freq=1)


# In[ ]:


history=model.fit([x_train,y_train[:,:-1]], y_train.reshape(y_train.shape[0],y_train.shape[1], 1)[:,1:] ,
          batch_size=32, # 700 / 32 ~= 22    # 8...512
          epochs = 3,
          verbose = 1,  #0,1,2
          validation_data=([x_validate,y_validate[:,:-1]], y_validate.reshape(y_validate.shape[0],y_validate.shape[1], 1)[:,1:]),
          callbacks = [tensorboard_callback, es],
          shuffle = False
          )


# In[ ]:





# In[56]:


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[ ]:




