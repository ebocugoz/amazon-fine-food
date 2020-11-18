import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
import string
import re
import seaborn as sns


class TextPreprocesser:
    
    preprocessers = []
    def __init__(self, preprocessers):
        self.preprocessers = preprocessers
        
    #Make all lowercase
    def to_lower(text):
        return text.lower()
        
    #Remove punctuations
    def remove_punctuation(text):
        return text.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
    
    def remove_html(text):
        return re.sub(re.compile('<.*?>'),' ',text)
    
    def remove_stopwords(text):
        new_tokens = []
        tokens = text.split(" ")
        stop_words = set(stopwords.words('english')) 
        for token in tokens:
            if token not in stop_words:
                new_tokens.append(token)
        return " ".join(new_tokens)
        
    #Lemmatization
    def lemmatize_text(text):
        lemmatizer = WordNetLemmatizer()
        tokens = []
        words = text.split(" ")
        for word in words:
            tokens.append(lemmatizer.lemmatize(word))

        return " ".join(tokens)
    
    
    def preprocess(self,df,column):
        """
        params:
          df: dataframe, which will update the text column with the new one
          column: name of the target column
        """
        for func in self.preprocessers:
            df[column] = df[column].apply(func)
        