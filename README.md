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

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
