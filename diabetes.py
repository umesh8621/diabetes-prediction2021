import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('diabetes.csv') #Loading the dataset
data.head(10)
data.tail()
data.shape
data.groupby('Outcome').size()
data.describe()
data.groupby('Outcome').hist(figsize=(9, 9))

data.isnull().any()
data.corr()

corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


from numpy import nan
data['Glucose']=data['Glucose'].replace(0,np.nan)
data['BloodPressure']=data['BloodPressure'].replace(0,np.nan)
data['SkinThickness']=data['SkinThickness'].replace(0,np.nan)
data['Insulin']=data['Insulin'].replace(0,np.nan)
data['BMI']=data['BMI'].replace(0,np.nan)

data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
data['BMI'].fillna(data['BMI'].median(), inplace=True)

data.isnull().any()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

X = data.drop(['Outcome'],axis=1)
y = data['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

print("X_train size:", X_train.shape)
print("y_train size: ",y_train.shape,"\n")
print("X_test size:", X_test.shape)
print("y_test size:",y_test.shape)


#standard scaling

sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

sc
X_train
X_test

# Outlier detection using boxplots
plt.figure(figsize= (20,15))
plt.subplot(4,4,1)
sns.boxplot(X['Pregnancies'])

plt.subplot(4,4,2)
sns.boxplot(X['Glucose'])

plt.subplot(4,4,3)
sns.boxplot(X['BloodPressure'])

plt.subplot(4,4,4)
sns.boxplot(X['SkinThickness'])


plt.subplot(4,4,5)
sns.boxplot(X['Insulin'])

plt.subplot(4,4,6)
sns.boxplot(X['BMI'])

plt.subplot(4,4,7)
sns.boxplot(X['DiabetesPedigreeFunction'])

plt.subplot(4,4,8)
sns.boxplot(X['Age'])



X['Pregnancies']=X['Pregnancies'].clip(lower=X['Pregnancies'].quantile(0.05), upper=X['Pregnancies'].quantile(0.95))
X['BloodPressure']=X['BloodPressure'].clip(lower=X['BloodPressure'].quantile(0.05), upper=X['BloodPressure'].quantile(0.95))
X['SkinThickness']=X['SkinThickness'].clip(lower=X['SkinThickness'].quantile(0.05), upper=X['SkinThickness'].quantile(0.95))
X['Insulin']=X['Insulin'].clip(lower=X['Insulin'].quantile(0.05), upper=X['Insulin'].quantile(0.95))
X['BMI']=data['BMI'].clip(lower=X['BMI'].quantile(0.05), upper=X['BMI'].quantile(0.95))
X['DiabetesPedigreeFunction']=X['DiabetesPedigreeFunction'].clip(lower=X['DiabetesPedigreeFunction'].quantile(0.05), upper=X['DiabetesPedigreeFunction'].quantile(0.95))
X['Age']=X['Age'].clip(lower=X['Age'].quantile(0.05), upper=X['Age'].quantile(0.95))



# Lets visualise the boxplots after imputing the outliers
plt.figure(figsize= (20,15))
plt.subplot(4,4,1)
sns.boxplot(X['Pregnancies'])

plt.subplot(4,4,2)
sns.boxplot(X['Glucose'])

plt.subplot(4,4,3)
sns.boxplot(X['BloodPressure'])

plt.subplot(4,4,4)
sns.boxplot(X['SkinThickness'])


plt.subplot(4,4,5)
sns.boxplot(X['Insulin'])

plt.subplot(4,4,6)
sns.boxplot(X['BMI'])

plt.subplot(4,4,7)
sns.boxplot(X['DiabetesPedigreeFunction'])

plt.subplot(4,4,8)
sns.boxplot(X['Age'])

print(X.columns)

# As we can see, there are still outliers in columns Skin Thickness and Insulin. Lets try manipulating the percentile values.
X['SkinThickness']=X['SkinThickness'].clip(lower=X['SkinThickness'].quantile(0.07), upper=X['SkinThickness'].quantile(0.93))
X['Insulin']=X['Insulin'].clip(lower=X['Insulin'].quantile(0.21), upper=X['Insulin'].quantile(0.80))
plt.figure(figsize= (20,15))
plt.subplot(2,2,1)
sns.boxplot(X['SkinThickness'])
plt.figure(figsize= (20,15))
plt.subplot(2,2,2)
sns.boxplot(X['Insulin'])






f, ax = plt.subplots(figsize=(20, 10))
corr = X.corr("pearson")
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax,annot=True)





from sklearn.ensemble import AdaBoostClassifier
rfc=AdaBoostClassifier()
rfc.fit(X_train,y_train)
predicted= rfc.predict(X_test)
print("Predicted Value:", predicted)

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, predicted))



# Creating a pickle file for the classifier
import pickle
filename = 'diabetes.pkl'
pickle.dump(rfc, open(filename, 'wb'))

