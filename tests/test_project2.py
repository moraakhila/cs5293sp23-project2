import project2
import pandas as pd
import pytest

def test_data():
	df,ing = project2.data()
	assert len(df)>0
	assert len(ing)>0
	assert isinstance(df,pd.DataFrame)
	assert isinstance(ing,list)

def test_vectorize():
	df = pd.DataFrame()
	ing = ['rice','bananas', 'salt', 'eggs', 'milk']
	ingredients = ['rice','salt']
	res = project2.vectorize(df,ing,ingredients)
	assert res.empty == False

def test_knn():
    dataf,ing = project2.data()
    vec_df = project2.vectorize(dataf.head(100),ing,['rice','salt'])
    res = project2.knn(dataf.head(100),vec_df)
    assert res != None

def test_display():
    dataf, ing = project2.data()
    vec_df = project2.vectorize(dataf.head(100), ing, ['rice','salt'])
    mid, cuisine_predict, cuisine_proba = project2.knn(dataf.head(100), vec_df)
    d = project2.display(mid, cuisine_predict, cuisine_proba, vec_df.iloc[-1].tolist(), vec_df.values.tolist(), dataf, int(5))
    assert d!=None
