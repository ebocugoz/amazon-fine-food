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
Here as score 0 refers to 1, 1->2, ..., 4->5
### Naive Bayes Confusion Matrix

![Alt text](https://github.com/ebocugoz/amazon-fine-food/blob/main/results/NB_cm.png?raw=true "Naive Bayes Confusion Matrix")

### Support Vector Machine Confusion Matrix

![Alt text](https://github.com/ebocugoz/amazon-fine-food/blob/main/results/SVM_cm.png?raw=true "Naive Bayes Confusion Matrix")

### Logistic Regression(Multiclass) Confusion Matrix

![Alt text](https://github.com/ebocugoz/amazon-fine-food/blob/main/results/LR_cm.png?raw=true "Naive Bayes Confusion Matrix")

### RNN Confusion Matrix

![Alt text](https://github.com/ebocugoz/amazon-fine-food/blob/main/results/RNN_cm.png?raw=true "Naive Bayes Confusion Matrix")

## Future Work
* Since the data is imbalanced we can try resampling the data (downscaling the 5s or upscaling 1-2-3-4)
* Training multiple models then ensemble
* Hyperparameter tuning
