
# Python Data Manipulation on a Titanic dataset  

 

## PROJECT DESCRIPTION 


## REQUIREMENT FOR THE CT7201 ASSESSMENT(Data Science Masters Class Set 2022/2023)

To develop a python notebook to analyse a titanic dataset using commented and correctly referenced python codes and libraries. Curating the questions below, we were able to complete this assessment in a well detailed manner:


What was our motivation? 

       To learn how to analyse data with python codes and libraries  

Why did we build this project? 

       To predict the survival rate of passengers of the titanic 

What problem did we solve? 

        By using some regression and classification models we were able to predict the survival rate with a 85% success rate 

What did we learn? 

       We learnt that certain factors influenced the survival of the passengers in the titanic like family size, gender, age, class and fare 

What makes our project stand out? 

    It is important to note the use of python libraries and functions to analyse the dataset. Several models were employed and the machine learning model with best prediction rate was about 85% to arrive at a good conclusion of prediction. The same classifier models were used with different independent variables and accuracies were compared to better understand what variables influenced the survival rate of passengers on the Titanic.  

 

The above questions were then answered with well detailed python codes under the following topics: 

>Factors determined passengers survival in the titanic 

>Factors that reduced the chances of survival of certain passengers 



 

TABLE OF CONTENT 

>Topics covered

>Titanic dataset picked by Team 8 

>Description of variables in the titanic dataset

>Python Libraries imported and their uses

>Set up tasks taken to get our codes to run

>Splitting of the Titanic dataset into test and train sets 

>Cleaning the dataset to remove any data not needed and correct any possible misspellings

>Data visualization of answers to the questions listed in the Project Description, using Seaborn library and Matplotlib library and other libraries 

>Classification techniques employed to create predictive models of the titanic dataset 

 

CONTENT

Topics covered:

>Python, Python Editors, Modules 

>Object oriented coding 

>Cleaning and manipulation of the dataset

>Data analysis 

>Feature engineering 


The Titanic dataset picked by Team 8: https://www.kaggle.com/c/titanic/data

Description of variables in the Titanic Dataset:

Pclass: Passenger classes based on fares paid and a possible refelction of their economic standings
1st = Upper
2nd = Middle
3rd = Lower
Fare =  Ticket charge per passenger
Age = Age of each passenger of the Titanic
Sibsp = dataset defines family relations : Siblings/Spouse
Sibling = Sister, Brother, Step-sister, Stepbrother
Parch : defines family relations
Child = person under 10
Teen = person between age 10 and 18 years
Embarked = Ports of entry of passengers namely Queenstown(Q), Cherbourg(C), Southampton(S)
Cabin = Cabin numbers of the passengers



Libraries imported were Pandas, NumPy, Math, Matplotlib, Seaborn, Sklearn, Collections

1. Pandas library was imported because it is a high level data manipulation tool needed to understand and visualise the structure of the dataset 

2. NumPy is a library for python which supports large arrays, so it was important in this assessment

3. Matplotlib was important to aid visualizations of the dataset as histograms, piecharts, scatterplots, bar plots

4. Seaborn library was imported as it is useful in showing individual feature details of variables important to the assessment

5. Several functions was Sklearn(Scikit learn) library was imported because it is used to model datasets as clusters and also for regression analysis, which helps to predict outcomes of the dataset

6. Isnan function is imported from Math module to check for missing values

7. Counter function is imported from Collections module to count the items in an iterable list


Different Python functions were used to explore the data and some after they were run had future warnings. Using a different python function in such instances produced the same results but no warnings as seen when the swarmplot function(from the seaborn package) was replaced with the stripplot function to show 'Survivals for Age and Pclass'. Also seen when 'Variance of fare calculated using the define function' gave the same output as 'Variance of fare calculated using numpy function'. This specifically illustrates how two different python functions can be used to manipulate the same set of variables and still give the same output(or visualization). Some python functions were found to print output faster than some did, even though the outputs were the same. This knowledge would help in time-sensitive data analysis work

Setup tasks to ensure python codes ran include installation of conda, python version 3.9, jupyter, Visual code. All the conda environments were installed before the pip function was called in Anaconda prompt. This includes the following codes:

1. conda env list

2. conda install Python pip

3. conda install Python = 3.9

4. conda install jupyter

5. conda install numpy


The Titanic dataset was split into test and train sets so that the latter dataset can be used to train the machine learning model. The test set is used to test the model to check the accuracy of the machine model predictions

The first step in cleaning this dataset was dropping columns which were not so important to our analysis. For example, in the ‘Embarked’ column, missing values were replaced with the mode values using the approriate python codes

Data visualization of answers to the questions listed in the Project Description, using Seaborn library and Matplotlib library and other libraries can be seen in our submitted pdf assessment file 

Some classification techniques were employed to create predictive models of the titanic dataset like the Random forest classifier which provided at good model accuracy of 0.86 of the train dataset and Logistic regression classifier which gave a good prediction of 0.85 of the train dataset.

Feature engineering was used to create Classes like 'Family' and 'Person'. For example, 'Person' Class was used to store values of passengers traveling with family.This helped to simplify our data transformation and improve the accuracy of machine models used in this assessment. 'Family' Class was useful to calculate the percentage of adult males and females who survived. Dictionaries like 'family_dict' was created to store values of family members and 'family_group_dict' was used to show the number of families in the titanic, which was 170

 

 

TEAM MEMBERS AND CREDITS 

Team Contribution from: Bassey Henshaw, Chanya Subasingha Arachchige, Zech-Enakhimion Ahmed and Oluwatoyin Odeniyi 


 

 

 
