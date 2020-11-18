import unittest
from textpreprocesser import * 
import pandas as pd

def createDf():
	items = [["This is, a TesT   fOr Text Process!! <br/>something<br/> apples going"]]
	df_test = pd.DataFrame(items, columns=["Text"])
	column = "Text"
	return df_test,column


class TestTextPreprocesser(unittest.TestCase):

	def test_to_lower(self):
		df_test,column = createDf()
		txtpr = TextPreprocesser([TextPreprocesser.to_lower])
		txtpr.preprocess(df_test,column)
		self.assertEqual(df_test.Text.values[0],"this is, a test   for text process!! <br/>something<br/> apples going")
	
	def test_remove_html(self):
		df_test,column = createDf()
		txtpr = TextPreprocesser([TextPreprocesser.remove_html])
		txtpr.preprocess(df_test,column)
		self.assertEqual(df_test.Text.values[0],"This is, a TesT   fOr Text Process!!  something  apples going")

	def test_remove_stopwords(self):
		df_test,column = createDf()
		txtpr = TextPreprocesser([TextPreprocesser.remove_stopwords])
		txtpr.preprocess(df_test,column)
		self.assertEqual(df_test.Text.values[0],"This is, TesT   fOr Text Process!! <br/>something<br/> apples going")

	def test_lemmatize_text(self):
		df_test,column = createDf()
		txtpr = TextPreprocesser([TextPreprocesser.lemmatize_text])
		txtpr.preprocess(df_test,column)
		self.assertEqual(df_test.Text.values[0],"This is, a TesT   fOr Text Process!! <br/>something<br/> apple going")


if __name__ == '__main__':
    unittest.main()