# Cuisine Predictor

In this project, the master list of all possible dishes, their ingredients, an identifier, and the cuisine for thousands of different dishes were given. If the foods were clustered by their ingredients, it can help restaurant change foods but keep ingredients constant. A display of clustered ingredients were presented and train a classifier to predict the cuisine type of new food. The data set used for this project is yummly.com(yummly.json). 

Author Details
•	Name: Akhila Mora
•	Email: akhila.mora@ou.edu
•	Student ID: 113531532

# Demo Video

# Getting Started
Below are the starting steps which needs to be done before starting the project:
* In Ubuntu, connect to the VM instance using the following command:
```
ssh -i [path-to-private-key] [username]@[instance-external-ip]
```
* Create a tree structure as shown below in VM instance:
* We need to have python installed in the instance. If not, install it using below command:
```
sudo apt-get install python3
```
* Json data is obtained from below url: https://oudatalab.com/cs5293sp23/projects/yummly.json

# Packages
Following are some of the packages used for this project: 
* argparse
* json
* pandas
* sklearn
* numpy

# Executing program
Below is the detailed explanation on how to run the project:
* Clone the project into your instance:
```
git clone https://github.com/moraakhila/cs5293sp23-project2
```
* Change the current working directory to cloned repository:
```
cd cs5293sp23-project2
```
* Create a virtual environment
```
pipenv install
```
* Activate virtual environment
```
pipenv shell
```
* Install necessary packages
```
  pipenv install nltk
  pipenv install pytest
  pipenv install sklearn
  pipenv install pandas
```
* Run the project using project2.py
```
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies" 
```
* By running the program using above command, it uses the ingredients passed from command line and gives a cuisine type. The output is in json format and below is the sample output:
![image](https://user-images.githubusercontent.com/113566461/234466349-4f980d9c-fcbe-442e-b2d1-2cbbf59e917f.png)

* pytests are used to check if the code passes all the test cases. Run pytests using below command:
```
pipenv run python -m pytest -v
```

## Functions
There is one file named project2.py which consists of all the logic to predict the cusine:
* project2.py: ‘argparse’ module is used to parse command line arguments ‘--N’ and ‘--ingredient’ with values. If any of the arguments is none, then an error message will be printed asking user to enter correct input. If both the arguments are not none, then it calls main(args) function.
* main(args):
   * This function takes arguments as parameter. The arguments are n(top closest meals) and ingredients to use for predicting. This function calls all the other functions one after the other. 
* data()
   * The json file(Yummly.json) located in the given url is converted into pandas dataframe. Based on the ‘id’ column, duplicate rows were deleted. Then, for the purpose of simplicity, all the ingredients were converted into lower case. Finally a list of unique ingredients were taken removing English stopwords and this function returns dataframe and list of ingredients.
* vectorize()
   * This function vectorizes dataframe based on ingredients. It takes three parameters (dataframe, ingredients and user_ing). It uses TfidfVectorizer with vocabulary ingredients and also sets binary parameter to ‘True’. It creates a binary vector for each cuisine representing presence or absence of an ingredient. Vectorized list is converted into dataframe and it is returned by the function.
* knn()
   * This function performs K-nearest neighbor classification on the input dataframe and vectorized dataframe. It returns indices of K-nearest neighbors, predicted cuisine label and probability of each cuisine label.
* display()
   * This function returns final output in the json format. It chooses top N  closest recipes and builds a dictionary with information on the cuisine, maximum cosine similarity score between input word vector and cuisine vectors, and top N closest recipes. It finally converts dictionary to JSON string and returns it. 






## Assumptions
For this project, I have made below assumptions:
•	All the ingredients in the json were converted into lowercase for simplicity. So while running the project, the ingredients needs to be entered in lowercase.
•	Assuming that the data is clean only stopwords were removed from the data.
•	The ingredients while running the project must be subset of json data.
•	This project can predict maximum of 10 closest id’s.

## Bugs

* The model may not predict with 100 % accuracy.
* If the passed ingredients were not present in the json data, then it may give wrong results.  

## Acknowledgments

* [Github Readme template](https://gist.github.com/DomPizzie/7a5ff55ffa9081f2de27c315f5018afc)
* [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [JSON](https://towardsdatascience.com/how-to-convert-json-into-a-pandas-dataframe-100b2ae1e0d8)
* [KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
