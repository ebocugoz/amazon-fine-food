import unittest
from modelsk import * 
import pandas as pd

def createDf():
	items = [["This is, a TesT   fOr Text Process!! <br/>something<br/> apples going",5]]*10
	df_test = pd.DataFrame(items, columns=["Text","Score"])
	column = "Text"
	return df_test["Text"],df_test["Score"]

class TestTextModelSK(unittest.TestCase):

	def test_get_train_test(self):
		X,y = createDf()
		mymodel = ModelSK(MultinomialNB(),X,y)
		X_train, X_test, y_train, y_test = mymodel.get_train_test(X,y,test_size = 0.2)
		self.assertEqual((len(X_train),len(y_train),len(X_test),len(y_test)),(8,8,2,2))

	def test_cross_val_train(self):
		X,y = createDf()
		mymodel = ModelSK(MultinomialNB(),X,y)
		mymodel.build_pipeline()
		mymodel.cross_val_train(scoring="f1_weighted",n_splits=3,verbose = False)
		scores = mymodel.cv_pred['test_score']
		self.assertEqual(len(scores),3)

if __name__ == '__main__':
    unittest.main()