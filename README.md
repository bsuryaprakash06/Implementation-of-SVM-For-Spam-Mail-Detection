# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. **Data Loading and Preprocessing**  
   - Read the dataset (`spam.csv`), detect file encoding, and handle missing values.  
   - Extract features (`v2`) as message text and labels (`v1`) as spam/ham.

2. **Data Splitting**  
   - Split the dataset into training and testing sets using `train_test_split` (e.g., 80% training, 20% testing).

3. **Feature Extraction**  
   - Convert text data into numerical features using `CountVectorizer`.  
   - Fit on the training data and transform both training and test sets.

4. **Model Training and Evaluation**  
   - Train the Support Vector Machine (`SVC`) model on the training data.  
   - Predict on the test data and evaluate performance using accuracy score, confusion matrix, and classification report.


## Program:
```python
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: B Surya Prakash
RegisterNumber:  212224230281
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()

data.isnull().sum()

data.info()

X=data["v2"].values
y=data["v1"].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_train=cv.fit_transform(X_train)
X_test=cv.transform(X_test)
X_train

X_test

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Name:B Surya Prakash")
print("Reg No:212224230281")
accuracy

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix

class_report=metrics.classification_report(y_test,y_pred)
print(class_report)
```

## Output:
<img width="1234" height="35" alt="image" src="https://github.com/user-attachments/assets/fed565f7-632b-499b-a170-9d79c6f32436" />

<img width="1231" height="217" alt="image" src="https://github.com/user-attachments/assets/450b775e-306d-49dc-a6b0-6453d55c6329" />

<img width="1220" height="145" alt="image" src="https://github.com/user-attachments/assets/1a6fc68c-870c-441a-8926-c3a3e4ce99e1" />

<img width="1182" height="254" alt="image" src="https://github.com/user-attachments/assets/a016ce24-da54-46a2-a1de-cc4384a15407" />

<img width="1192" height="152" alt="image" src="https://github.com/user-attachments/assets/9ff608d2-7829-4c7d-bc8f-9e615deb99e4" />

<img width="998" height="40" alt="image" src="https://github.com/user-attachments/assets/b5ddf24a-23cf-43f7-8c53-b1b1f813bd89" />

<img width="999" height="42" alt="image" src="https://github.com/user-attachments/assets/ae6f1cf0-a03c-487d-bcf3-487117ea04dc" />

<img width="1003" height="30" alt="image" src="https://github.com/user-attachments/assets/69765e4b-83b6-4880-9d9b-400e2d0d1a42" />

<img width="994" height="73" alt="image" src="https://github.com/user-attachments/assets/509c6653-5046-4f02-a4c1-2ab9401c9da7" />

<img width="995" height="49" alt="image" src="https://github.com/user-attachments/assets/d64533f2-68db-4a72-9f1d-8c9712094317" />

<img width="993" height="162" alt="image" src="https://github.com/user-attachments/assets/350e5247-106b-402b-b4f5-44f5aa9c756d" />



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
