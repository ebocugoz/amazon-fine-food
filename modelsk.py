from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

class ModelSK:
    def __init__(self, model,X,y):
        """
        model: sklearn model
        X: Text column
        y: Score column
        """
        self.model = model
        self.X = X
        self.y = y
        self.pipe = None
        self.cv_pred = None
        self.best_model = None

    def get_train_test(self,X,y,splittype="stratified",test_size = 0.33):
      """
      This function returns splitted train test pairs for X and y. 
      params
        X: Text column
        y: Score column
        splittype(string): The type of train test split, it can be stratified or methods like 
                    upsampling/downsampling to deal with imbalanced data
      outs:
       X_train, X_test, y_train, y_test
      """
      if splittype == "stratified":
        #Stratified train/test split since data is skewed
        X_train, X_test, y_train, y_test = train_test_split(
                          self.X, self.y, test_size= test_size, random_state=42,stratify=self.y)
        return  X_train, X_test, y_train, y_test
      else:
        print("Unknown split type")
        return None

    def build_pipeline(self,ngram_range = (1,2)):
      """
      Buids a pipeline
      1. CountVectorizer: Converts texts to a matrix of token counts (if you want to include 2 ngram set to (1,2) )
      2. TfidfTransformer: Transform the count matrix to a normalized tf-idf representation
      3. Model: Selected model to predict score
      """
      pipe = Pipeline([('CVec', CountVectorizer()),
                     ('Tfidf', TfidfTransformer()),
                     ('model', self.model)])
      
      self.pipe = pipe

    def cross_val_train(self,scoring,n_splits=5,verbose = True):
      """
      Trains the model by using cross validation
      params:
        scoring(string): scoring method to evaluate the model
        n_splits(int): number of splits for the cross validation
      """
      skf = StratifiedKFold(n_splits=n_splits) #Stratified split
      cv_pred = cross_validate(self.pipe,self.X, self.y, cv=skf,scoring=(scoring), n_jobs=-1, verbose =verbose)
      self.cv_pred=cv_pred



    
    def plot_confusion_matrix(self,n_splits=5,normalize=True):
      """
      Plots a confusion matrix from the model
      params:
        n_splits(int)= number of splits for the cross validation
        normalize(boolean) = If true prints a normalized matrix
      """
      skf = StratifiedKFold(n_splits=n_splits)
      y_pred = cross_val_predict(self.pipe, self.X, self.y, cv=skf)
      conf_mat = confusion_matrix(self.y, y_pred)
      if normalize:
        conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
      sns.heatmap(conf_mat, annot=True)


      def parameter_tuning(self):
        param_grid = [
                      {'model' : [self.model],
                      'classifier__penalty' : ['l1', 'l2'],
                      'classifier__C' : np.logspace(-4, 4, 20),
                      'classifier__solver' : ['liblinear']},
                      ]
      clf = GridSearchCV(self.pipe, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)

      

        


