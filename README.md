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


# % of rows missing in each column

for column in train_df.columns:
    percentage = train_df[column].isnull().mean()
    print(f'{column}: {round(percentage*100,4)}%')

#####New Additions 

###insert this plot at the begining of the data when we still had missing values, this is a graphical presentation of the variable where there is missing data in the data set

plt.subplots(figsize=(10,6)) #the figsize help zoom our plot to give a better visible view
sns.heatmap(train_df.isnull(), cbar=False, cmap="YlGnBu_r")
plt.show()




categorical_variable = ['Survived', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']

'Our desired number of plot row and column size'
categorical_plot_nrows = 2
categorical_plot_ncols = 3

fig, axs = plt.subplots(categorical_plot_nrows, categorical_plot_ncols, figsize=(categorical_plot_ncols*3.5, categorical_plot_nrows*3))

for r in range(0, categorical_plot_nrows):
    for c in range(0, categorical_plot_ncols):
        
        i = r*categorical_plot_ncols+c
        ax = axs[r][c]
        sns.countplot(train_df[categorical_variable[i]], hue=train_df["Survived"], ax=ax)
        ax.set_title(categorical_variable[i], fontsize=12, fontweight="bold")
        ax.legend(title="Survived", loc='upper center')

plt.tight_layout()




sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass")
plt.show()



sns.barplot(x='Sex', y='Survived', hue='Pclass', data=train_df)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Pclass and Sex")
plt.show()



sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Embarked Port")
plt.show()


sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=train_df)
plt.ylabel("Survival Rate")
plt.title("Survival as function of Embarked Port & Class")
plt.show()


sns.countplot(x='Embarked', hue='Pclass', data=train_df)
plt.title("Count of Passengers as function of Embarked Port")
plt.show()


sns.boxplot(x='Embarked', y='Age', data=train_df)
plt.title("Age distribution as function of Embarked Port")
plt.show()


sns.boxplot(x='Embarked', y='Fare', data=train_df)
plt.title("Fare distribution as function of Embarked Port")
plt.show()


fig, ax = plt.subplots(figsize=(13, 7))
sns.violinplot(x="Pclass", y="Age", hue='Survived', data=train_df,
               split=True, bw=0.05, palette=swarm_color, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()



swarm_color = ["red", "green"]
fig, ax = plt.subplots(figsize=(13, 7))
sns.swarmplot(x='Pclass', y='Age', hue='Survived', split=True,
              data=train_df, palette=swarm_color, size=7, ax=ax)
plt.title('Survivals for Age and Pclass ')
plt.show()




g = sns.factorplot(x="Pclass", y="Age", hue="Survived", col="Sex", data=train_df,
                   kind="swarm", split=True, palette=swarm_color, size=7, aspect=.9, s=7)




y="Age", hue="Survived", col="Sex", data=train_df, kind="violin", split=True, bw=0.05, palette=swarm_color, size=7, aspect=.9, s=7




## Corrections and Additions
i forgot to drop this Age plot

train_df["Age"].plot()


sns.heatmap(correlation, annot=True) #the heat plot showing the correlation matrix


Building a Predictive Model to predict survivor in the titanic


#Visualizing the shape of the data


plt.figure(figsize=(20,4))     #for plotting a figure
for index, (image, label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):#this is taking a sample of the images that we have loaded ifor training purposes
    plt.subplot(1,5, index + 1)
    plt.imshow(np.reshape(image, (8,8)),cmap=plt.cm.gray)
    plt.title('Training: %i\n' %label, fontsize=20)
    
    
    
 #Splitting our data in Train data set and Test data Set, so that the  training data is use for train the model and the test data set is use to test the model
x = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = train_df["Survived"]

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=2)


# Formating the data to fit the predictive model
train_df.loc[train_df['Survived'] == 0, 'Survived'] = 'Deceased'
train_df.loc[train_df['Survived'] == 1, 'Survived'] = 'Survived'
train_df['Survived'].head()


train_df.loc[train_df['Pclass'] == 'First Class', 'Pclass'] = 1
train_df.loc[train_df['Pclass'] == "Second Class", 'Pclass'] = 2
train_df.loc[train_df['Pclass'] == 'Third Class', 'Pclass'] = 3
train_df['Pclass'].head()


train_df.loc[train_df['Embarked'] == 'C', 'Embarked'] = 1
train_df.loc[train_df['Embarked'] == 'Q', 'Embarked'] = 2
train_df.loc[train_df['Embarked'] == 'S', 'Embarked'] = 3

train_df.loc[train_df['Sex'] == "male", 'Sex'] = 1
train_df.loc[train_df['Sex'] == "female", 'Sex'] = 2


print(x)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

# fitting the model for Logistic Regression analysis
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_test)
print(predictions) #the predicted values


score = logisticRegr.score(x_test, y_test)
print(score)

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9, 9))
sns.heatmap(confusion_matrix,annot=True, fmt='.3f', linewidths=.5, square=True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Model Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);

