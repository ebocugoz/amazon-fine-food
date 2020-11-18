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
  * textpreprocesser.py : Class with functions for preprocessing the text data. 
  * textpreprocesser_tester.py : Unit tests for textpreprocesser.py
  * modelsk.py : Class to build machine learning pipelines by using sklearn.  
  * modelsk_tester.py : Unit tests for modelsk.py
  * modelkeras.py : Class to build neural network models by using keras.
  * keras_plotters.py : Functions to plot learning curve for keras model.
  

  
