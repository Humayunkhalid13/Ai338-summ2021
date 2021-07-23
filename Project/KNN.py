importnumpyas np
import pandas as pd
importmatplotlib.pyplotasplt
import seaborn assns
import warnings
warnings.filterwarnings('ignore')
train =pd.read_csv("D:\Titanic/train.csv")
test =pd.read_csv("D:\Titanic/test.csv")

#Preparing the training data
train.describe(include="all")
										
												
print(train.columns)
train.sample(5)
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
dtype='object')
train.describe(include = "all")

print(pd.isnull(train).sum())
sns.barplot(x="Sex", y="Survived", data=train)

##printing all the percentages of males vs. females that were survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize =True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize =True)[1]*100)
Percentage of females who survived: 74.20382165605095
Percentage of males who survived: 18.890814558058924

sns.barplot(x="Pclass", y="Survived", data=train)

#printing percentage of people that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize =True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize =True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize =True)[1]*100)
Percentage of Pclass = 1 who survived: 62.96296296296296
Percentage of Pclass = 2 who survived: 47.28260869565217
Percentage of Pclass = 3 who survived: 24.236252545824847

sns.barplot(x="SibSp", y="Survived", data=train)

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize =True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize =True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize =True)[1]*100)
Percentage of SibSp = 0 who survived: 34.53947368421053
Percentage of SibSp = 1 who survived: 53.588516746411486
Percentage of SibSp = 2 who survived: 46.42857142857143

sns.barplot(x="Parch", y="Survived", data=train)
plt.show()

train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] =pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] =pd.cut(test["Age"], bins, labels = labels)

#draw a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculating the percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize =True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize =True)[1]*100)
#drawing a bar plot of Survived vs. CabinBool
sns.barplot(x="CabinBool", y="Survived", data=train)
plt.show()
Percentage of CabinBool = 1 who survived: 66.66666666666666
Percentage of CabinBool = 0 who survived: 29.985443959243085

test.describe(include="all")
													
train =train.drop(['Cabin'], axis = 1)
test =test.drop(['Cabin'], axis = 1)
train =train.drop(['Ticket'], axis = 1)
test =test.drop(['Ticket'], axis = 1)
print("Number of people embarking in Southampton (S):")
southampton= train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg= train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown= train[train["Embarked"] == "Q"].shape[0]
print(queenstown)

train =train.fillna({"Embarked": "S"})combine = [train, test]

#extract a title for each Title within the train and test datasets
for dataset in combine:
    dataset['Title'] =dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])
		
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping= {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()
mr_age= train[train["Title"] ==1]["AgeGroup"].mode() #Young Adult
miss_age= train[train["Title"] ==2]["AgeGroup"].mode() #Student
mrs_age= train[train["Title"] ==3]["AgeGroup"].mode() #Adult
master_age= train[train["Title"] ==4]["AgeGroup"].mode() #Baby
royal_age= train[train["Title"] ==5]["AgeGroup"].mode() #Adult
rare_age= train[train["Title"] ==6]["AgeGroup"].mode() #Adult

age_title_mapping= {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}

#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})
#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})

for x in range(len(train["AgeGroup"])):
if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] =age_title_mapping[train["Title"][x]]

for x in range(len(test["AgeGroup"])):
if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] =age_title_mapping[test["Title"][x]]
In [32]:
age_mapping= {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age feature for now, might change
train =train.drop(['Age'], axis = 1)
test =test.drop(['Age'], axis = 1)
In [33]:
train =train.drop(['Name'], axis = 1)
test =test.drop(['Name'], axis = 1)
In [34]:
sex_mapping= {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()
embarked_mapping= {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()
											
for x in range(len(test["Fare"])):
ifpd.isnull(test["Fare"][x]):
pclass= test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] ==pclass]["Fare"].mean(), 4)

#map Fare values into groups of numerical values
train['FareBand'] =pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] =pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#drop Fare values
train =train.drop(['Fare'], axis = 1)
test =test.drop(['Fare'], axis = 1)
train.head()
test.head()

fromsklearn.model_selectionimporttrain_test_split

predictors =train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val=train_test_split(predictors, target, test_size= 0.22, random_state= 0)
importnumpyas np
fromsklearn.model_selectionimportKFold
fromsklearn.model_selectionimportcross_val_score
fromsklearn.neighborsimportKNeighborsClassifier

knn=KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred=knn.predict(x_val)
acc_knn=round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)
importnumpyas np
fromsklearnimport metrics
rounding =np.round(y_pred_knn)
acc_knn=metrics.accuracy_score(y_val,rounding)  
acc_knn
kf_neighbors=KFold(n_splits=6,shuffle=True)
cv_neighbors=cross_val_score(knn_model, predictors, target, cv=kf_neighbors)
np.mean(cv_neighbors)
ids = test['PassengerId']
predictions =knn_model.predict(test.drop('PassengerId', axis=1))

output =pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('D:\Titanic/submissionsKNN11.csv', index=False)

