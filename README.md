
# Python Data Manipulation on a Titanic dataset  

 

## PROJECT DESCRIPTION 


## REQUIREMENT FOR THE CT7201 ASSESSMENT

To develop a python notebook to analyse a titanic dataset using commented and correctly referenced python codes and libraries. Curating the questions below, we were able to complete this assessment in a well detailed manner:


What was our motivation? 

       To learn how to analyse data with python codes and libraries  

Why did we build this project? 

       To predict the survival rate of passengers of the titanic 

What problem did we solve? 

        By using some regression and classification models we were able to predict the survival rate with a 94% success rate 

What did we learn? 

       We learnt that certain factors influenced the survival of the passengers in the titanic like family size, gender, age, class and fare 

What makes our project stand out? 

    It is important to note that we built several machine models and we picked the machine model with best prediction rate of 94% to arrive at our conclusion. We used the same classifier models on different independent variables and compared their accuracies, to understand what variables influenced the survival rate of passengers on the Titanic.  

 

The above questions were then answered with well detailed python codes under the following topics: 

>Factors determined passengers survival in the titanic 

>Factors that reduced the chances of survival of certain passengers 



 

TABLE OF CONTENT 

>Libraries imported and their uses

>Titanic dataset picked by Team 8  

>Set up tasks taken to get our codes to run

>Splitting of the Titanic dataset into test and train sets 

>Cleaning the dataset 

>Data visualization of answers to the questions listed in the Project Description, using Seaborn library and Matplotlib library and other libraries 

>Classification techniques employed to create predictive models of the titanic dataset 

 

TOPICS COVERED 

>Python, Python Editors, Modules 

>Object oriented coding 

>Cleaning and manipulation of the dataset

>Data analysis 

>Feature engineering 

Variables in the Titanic Dataset

Pclass: Passenger classes based on fares paid and a possible refelction of their economic standings
1st = Upper
2nd = Middle
3rd = Lower

Fare =  Ticket charge per passenger
Age = Age of each passenger of the Titanic
Sibsp = dataset defines family relations : Siblings/Spouse
Sibling = Sister, Brother, Step-sister, Stepbrother
Spouse = Husband and Wife
Parch : defines family relations
Parent = Mother and Father
Child = Daughter, Son, Step-daughter and Step-son
Embarked = Ports of entry of passengers namely Queenstown(Q), Cherbourg(C), Southampton(S)
Cabin = Cabin numbers of the passengers



CONTENT 

Libraries imported were Pandas, NumPy, Matplotlib, Seaborn, Sklearn, Plotly. The Titanic dataset picked by Team 8: https://www.kaggle.com/c/titanic/data. Pandas library was imported because it is a high level data manipulation tool needed to understand and visualise the structure of the dataset. NumPy is a library for python which supports large arrays, so it was important in this assessment. Matplotlib was important to aid visualizations of the dataset as histograms, piecharts, scatterplots, bar plots.Seaborn library was imported as it is useful in showing individual feature details of variables important to the assessment. Sklearn/Scikit learn library was imported because it is used to model datasets as clusters and also for regression analysis, which helps to predict outcomes of the dataset.

The Titanic dataset was split into test and train sets so that the latter dataset can be used to train the machine learning model. The test set is used to test the model to check the accuracy of the machine model predictions. 

The first step in cleaning this dataset was dropping columns which were not so important to our analysis. In the ‘Embarked’ column, missing values were replaced with the mode values using the approriate python codes.

Data visualization of answers to the questions listed in the Project Description, using Seaborn library and Matplotlib library and other libraries can be seen in our submitted pdf assessment file 

Some classification techniques were employed to create predictive models of the titanic dataset n the barplot showing survived versus fatalities of the titanic dataset

Feature engineering was used to create Classes like Family and Person. This was to simplify our data transformation and improve the accuracy of machine models used in this assessment.

 

 

TEAM MEMBERS AND CREDITS 

Team Contribution from: Bassey Henshaw, Chanya Subasingha Arachchige, Oluwatoyin Odeniyi and Zech-Enakhimion Ahmed.


 

 

 
