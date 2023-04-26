import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
import json
import requests

def main(args):
	n = args.N
	ingredients = args.ingredient
	df, ing = data()
	df2 = vectorize(df, ing, ingredients)
	mid, cuisine_predict, cuisine_proba = knn(df, df2)
	d = display(mid, cuisine_predict, cuisine_proba, df2.iloc[-1].tolist(), df2.values.tolist(), df, int(n))


def data():
	response = requests.get("https://oudatalab.com/cs5293sp23/projects/yummly.json")
	data = json.loads(response.content)
	dataframe = pd.DataFrame(data)
	dataframe = dataframe.drop_duplicates(['id'])
	for i in range(len(dataframe)):
		lower = []
		for j in dataframe.at[i, 'ingredients']:
			lower.append(j.lower())
		dataframe.at[i, 'ingredients'] = lower
	ing_list = dataframe['ingredients'].tolist()
	flat_list = [item for sublist in ing_list for item in sublist]
	unique_list = list(np.unique(flat_list))
	ing = [word for word in unique_list if word not in stopwords.words('english')]
	return dataframe, ing

def vectorize(df, ing, ingredients):
    v_list = []
    vectorizer = TfidfVectorizer(vocabulary=ing,binary=True)
    vectorizer.fit(ing)
    for i in range(len(df)):
        my_vector = vectorizer.transform([' '.join(df.at[i,'ingredients'])]).toarray()[0]
        my_vector[my_vector > 0] = 1
        v_list.append(list(my_vector))
    my_vector = vectorizer.transform([' '.join(ingredients)]).toarray()[0]
    my_vector[my_vector > 0] = 1
    v_list.append(list(my_vector))
    df2=pd.DataFrame(v_list)
    df2.columns=ing
    #print(df2.head(100))
    return df2

def knn(dataframe,df2):
    knn = KNeighborsClassifier(n_neighbors=10)
    df3 = df2.drop(df2.index[-1])
    knn_class = knn.fit(df3,dataframe['cuisine'])
    test=np.array(df2.iloc[-1].tolist()).reshape(1,-1)
    cuisine_proba = knn_class.predict_proba(test)
    cuisine_predict = knn_class.predict(test)
    classe = knn.classes_
    dist,mid = knn_class.kneighbors(test)
    return mid, cuisine_predict, cuisine_proba

def display(c_id, one_cuisine, cuisine, u_ing_list, v_list, df, N):
    id_list = df['id'].tolist()
    close_id = [id_list[i] for i in c_id[0]]
    close_vect = [v_list[i] for i in c_id[0]]
    l = []
    for v in close_vect:
        id_score = cosine_similarity([u_ing_list], [v])[0][0]
        l.append(round(id_score, 2))
    close_dict = [{'id':id_, 'score': float(score)} for id_, score in zip(close_id,l)]
    close_dict = sorted(close_dict, key = lambda x:x['score'], reverse = True)
    close_dict = close_dict[:N]
    d = {"cuisine": one_cuisine[0], "score": max(list(cuisine)), "closest": close_dict}
    d['score'] = max(d['score'].tolist())
    res = json.dumps(d, indent=4)
    print(res)
    return res

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--N",type = str, required = True, help = "Top N closest meals")
	parser.add_argument("--ingredient", action = 'append', type = str, required = True, help = "Ingredients list")
	args = parser.parse_args()
	ingredients = args.ingredient
	for i in ingredients:
		if str(i).isdigit() == False or len(i) == 0:
			break;
	if(args.N != None and args.ingredient != None):
		main(args)
	else:
		print("Please enter correct input!")
