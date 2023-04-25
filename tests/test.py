import project2
import pytest

def test_data():
	df,ing = data()
	assert len(df)>0
	assert len(ing)>0
	assert isinstance(df,pd.DataFrame)
	assert isinstance(ing,list)

