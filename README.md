# Group8
Group assignment for CT7201 Python Notebooks and Scripting (SEM2 - 2021/22)



###insert this plot at the begining of the data when we still had missing values, this is a graphical presentation of the variable where there is missing data in the data set
plt.subplots(figsize=(10,6)) #the figsize help zoom our plot to give a better visible view
sns.heatmap(train_df.isnull(), cbar=False, cmap="YlGnBu_r")


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
