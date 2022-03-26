# Group8
Group assignment for CT7201 Python Notebooks and Scripting (SEM2 - 2021/22)

###Importing the important libraries needed for Data Manipulation, classification and Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.datasets import load_digits
digits = load_digits()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


from sklearn.impute import SimpleImputer



print("\tData description")
print("*****************************************")
print("""•\tPassengerId: Id of every passenger.\n•\tSurvived: This feature have value 0 and 1. 0 for not survived and 1 for survived.\n•\tPclass: There are 3 classes: Class 1, Class 2 and Class 3.\n•\tName: Name of passenger.\n•\tSex: Gender of passenger.\n•\tAge: Age of passenger.\n•\tSibSp: Indication that passenger have siblings and spouse.\n•\tParch: Whether a passenger is alone or have family.\n•\tTicket: Ticket number of passenger.\n•\tFare: Indicating the fare.\n•\tCabin: The cabin of passenger.\n•\tEmbarked: The embarked category""")


data_file = "train.csv"
train_df = pd.read_csv(data_file)
train_df.head(10)
train_df.dtypes
train_df.shape



train_df.head(n=6)



### Transforming the survived column from integer to object

train_df.loc[train_df['Survived'] == 0, 'Survived'] = 'Deceased'
train_df.loc[train_df['Survived'] == 1, 'Survived'] = 'Survived'
train_df['Survived'].head()


train_df.loc[train_df['Pclass'] == 1, 'Pclass'] = 'First Class'
train_df.loc[train_df['Pclass'] == 2, 'Pclass'] = 'Second Class'
train_df.loc[train_df['Pclass'] == 3, 'Pclass'] = 'Third Class'
train_df['Pclass'].head()


train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 'Cherbourg'
train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 'Queenstown'
train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 'Southampton'


train_df.head()



###The average age of passengers?
train_df["Age"].mean()


### Median Age and Ticket fare price of Pax
train_df[["Age", "Fare"]].median()


train_df.describe(include="all")


###instead of using the predifined "describe function" we can combined aggregating statistics
train_df.agg(
{
    "Age": ['min', 'max', 'mean', 'median', 'skew'],
    "Fare": ['min', 'max', 'mean', 'median', 'skew'],
}
)


### counting by group
train_df.groupby("Pclass")["Pclass"].count()





train_df.isnull()  ### is any missing values in dataframe


train_df.isnull().any()  ### is any missing values across columns



train_df.isnull().sum().sum()   ###count of missing values of the entire dataframe


count_NAN = len(train_df) - train_df.count() ###count of missing values across columns
count_NAN

for i in range(len(train_df.index)):  ### count of missing values across rows
    print("NAN in row ", i , " : ", train_df.iloc[i].isnull().sum())


###counting of missing values of a particular column
train_df.Age.isnull().sum()


### count of missing values of column by group
'''count of missing values of column by group
 Because we dont havve missing values in sex all the missing values is coming from Age'''
 
train_df.groupby(['Sex'])['Survived'].apply(lambda x: x.isnull().sum())

# Replacing the na in age column with mean
train_df["Age"] = train_df["Age"].fillna(train_df["Age"].mean())
train_df["Age"].head()
#checking if there are any remaining missing value
train_df.Age.isnull().sum()
