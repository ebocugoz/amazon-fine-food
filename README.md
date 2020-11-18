# amazon-fine-food

### Goal :
Predict user scores(1-5) from their Reviews

## Required Libraries and Setting up the Envioronment 

* Used python evironment(anaconda)

pip install libraries
* scikit-learn
* Pandas
* NumPy
* os
* spacy
* nltk
* tensorflow, keras
* seaborn
* re

* Also need to download required nltk packages 


## Loading data
Data is avaialable in https://www.kaggle.com/snap/amazon-fine-food-reviews, you need to put the data in a folder named "archive" in the same directory.



## Files
* Python files :
  * Amazon_Food_Review.ipynb : Main notebook.


* Python files :
  * textpreprocesser.py : Class with functions for preprocessing the text data. 
  * textpreprocesser_tester.py : Unit tests for textpreprocesser.py
  * modelsk.py : Class to build machine learning pipelines by using sklearn.  
  * modelsk_tester.py : Unit tests for modelsk.py
  * modelkeras.py : Class to build neural network models by using keras.
  * keras_plotters.py : Functions to plot learning curve for keras model.
  
  
## Steps

* Loading & Exploring the data
* Preprocessing the data
  * Converting text to lower case
  * Removing html tags
  * Removing punctuations
  * Removing stop words
  * Lemmatization
* Creating Dictionary
  * Creating frequency dictionary
  * Get top **n** frequent words
  * Removing reviews longer than **k** words
* Building SKlearn Models
  * Naive Bayes
  * Support Vector Machine
  * Logistic Regression
* Building Keras Models
  * RNN 
  
## Results

![Alt text](/results/NB_cn.png?raw=true "Naive Bayes Confusion Matrix")

  
