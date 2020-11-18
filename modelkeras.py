import tensorflow as tf 
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model,Sequential
from keras.layers import Input, Masking, TimeDistributed, Dense, Concatenate, Dropout, LSTM, GRU, SimpleRNN, Bidirectional, Embedding, BatchNormalization
from keras.optimizers import Adam
from numpy import argmax
from sklearn.metrics import confusion_matrix



class ModelKeras:


    
    def __init__(self,X,y,maxlen):
        self.model = None
        self.X = X
        self.y = y
        self.pipe = None
        self.cv_pred = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.maxlen = maxlen
        self.METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            ]




    def tokenize_xy(X,y):
      """
      Tokenize the X and y according to previously created frequency dictionary
      """
      tokenizer = Tokenizer(num_words=len(dictionary_frequent))
      tokenizer.fit_on_texts(X)
      X = tokenizer.texts_to_sequences(X)
      X = sequence.pad_sequences(X, maxlen=self.maxlen)
      y = to_categorical(y)

      self.X = X
      self.y = y

    def tokenize_xy_train_test(X_train,X_test,y_train,y_test):
      """
      Tokenize the X and y train-test sets according to previously created frequency dictionary
      """
      X_train = tokenizer.texts_to_sequences(X_train)
      X_test = tokenizer.texts_to_sequences(X_test)

      X_train = sequence.pad_sequences(X_train, maxlen=self.maxlen)
      X_test = sequence.pad_sequences(X_test, maxlen=self.maxlen)

      y_train = to_categorical(y_train)
      y_test = to_categorical(y_test)
      self.X_train = X_train
      self.X_test = X_test

      self.y_train = y_train
      self.y_test = y_test

    def prepare_train_test(self,splittype="stratified"):
      """
      This function splits train and test, then 
      params: 
        splittype: the type of train test split
      """

      if splittype == "stratified":
        #Stratified train/test split since data is skewed
        X_train, X_test, y_train, y_test = train_test_split(
                          self.X, self.y, test_size=0.2, random_state=42,stratify=self.y)
      elif splittype == "none":
        X_train, X_test, y_train, y_test = train_test_split(
                          self.X, self.y, test_size=0.2, random_state=42)
      else:
        print("Unknown split type")
        return None
      
      self.maxlen = 348
      tokenize_xy(self.X,self.y)
      tokenize_xy_train_test(X_train,X_test,y_train,y_test)

 


    def construct_model(self, embedding_vecor_length = 64, dropout=0.2, output_size=6,lstm_hiddensize = 128): 
      input_length = self.maxlen  
      model = Sequential()
      model.add(Embedding(len(dictionary_frequent), embedding_vecor_length, input_length=input_length))
      model.add(Dropout(dropout))
      model.add(Bidirectional(LSTM(lstm_hiddensize)))
      model.add(Dropout(dropout))
      model.add(Dense(output_size, activation='softmax'))
      model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=self.METRICS)
      self.model = model
      print(model.summary())
    
    def fit(self,epochs=3,batch_size=64,validation_split=0.2,selfVal = False):
      if selfVal:
        return self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size,validation_split = validation_split)
      else:
        return self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size,validation_data=(self.X_test, self.y_test))
       

    def predict(self):
        return self.model.predict(self.X_test)

    def get_plot_confusion_matrix(self,y_predict,normalize=True):
      
      y_original_predict = argmax(y_predict, axis=1)
      y_original_test = argmax(self.y_test, axis=1)

      conf_mat = confusion_matrix(y_original_test, y_original_predict)

      if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
      sns.heatmap(conf_mat, annot=True)
      return conf_mat
        
    