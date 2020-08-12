# udacity-Seattle-Airbnb

## 1) Introduction
This project is part of the Udacity's Data Science Nanodegree program where I have chosen to write a blog post using AirBnB's Seattle Open Data, demonstrating a data-driven approach towards the understanding of how property features drive the pricing and popularity of Airbnb listings in Seattle.

## 2) Motivation of the project
Using Seattle's AirBnB data, and with the CRISP-DM standard as a guide (Cross Industry Process for Data Mining), we hope to gain more understanding on the Seattle rental market, specifically, 3 questions regarding:

#### What property features determine the listing price?

#### What property features determine the its popularity? (with reviews per month as proxy)

#### When is the most popular month to rent in Seattle?

Our approaches and findings will be documented in the python scripts, notebooks and in a written Medium blog post.

## 3) Libraries Used
This project is exclusively undertaken within the Anaconda enviroment using python 3.6 and the following libraries are used:
glob
numpy 
pandas
datetime
matplotlib
scipy.sparse (csr_matrix module)
sklearn.impute (SimpleImputer module)
sklearn.tree (DecisionTreeClassifier module)
sklearn.ensemble (AdaBoostClassifier module)
sklearn.feature_extraction.text (CountVectorizer module)
sklearn.preprocessing (inMaxScaler, StandardScaler modules)
sklearn.model_selection (train_test_split, GridSearchCV modules)
sklearn.metrics (make_scorer,fbeta_score, accuracy_score modules)

## 4) Files in the Repository (show the folder structure and describe each file)
#### AirBnB_Seattle-Submission_v2.ipynb
This notebook documents the workings in arriving at key learnings of the project.

#### helper_functions.py
This python file contains the functions used in the aforementioned notebook.

#### Jupyter Notebook in html + pdf format
There seems to be issue with loading the notebook in github, so I have included the html and pdf versions of the notebook:

AirBnB_Seattle-Submission_v3.html

AirBnB_Seattle-Submission_v3.pdf

## 5) Summary of the result
We have demonstrated that simple and straightforward data-driven investigation of the Seattle Airbnb market has provided us with insights may be helpful to anything keen to be involved in the property rental business.

We found that listing price and popularity of the properties are influenced by different combinations of attributes that cut across attributes concerning the property itself (rooms, amenities, etc) and that of the owner (experience, responsiveness, price set by host). 

This suggests our questions are distinct to one another, else, answering the fundamentally identical question would have resulted in the same top features (that are driving the accuracy of our models).

Further to this, we also show that the most popular month to rent in Seattle could be driven by some variable that is not part of our dataset. 

All the findings of the project can be found at the post available here.

https://medium.com/@taijiensing/understanding-airbnb-listings-in-seattle-9ff04b98c844

## 6) Acknowledgement | Attribution
Data sources exceeds the github file size quota but they are obtained from kaggle as below. Other references are cited within the notebook.

Seattle Airbnb Open Data:

https://www.kaggle.com/airbnb/seattle
