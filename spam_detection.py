import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # popular classification library for binary classification
from sklearn.metrics import accuracy_score   # performance matrix
df = pd.read_csv('mail_data.csv')
print(df)
data = df.where(pd.notnull(df),'')


data.head(10)   ##print top 10 lines of the dataset

data.info()
data.shape  ##rows and cols

data.loc[data['Category'] =='spam', 'Category',]=0
data.loc[data['Category']=='ham','Category',]=1

X=data['Message']
Y=data['Category']

print(X)
print(Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)  #training x and traing y same for x and y size of test is set to 0.2 i.e 80 % train and testing is 20%
#random state is the hyperparameter to get the consitent result. Due the spliting of the data clustering takes place hence random is used

print(X.shape)  #5572 total rows
print(X_train.shape)  # 80% of x shape go for training
print(X_test.shape)   #20% go  for testing

print(Y.shape)  #5572 total rows
print(Y_train.shape)  # 80% of Y shape go for training
print(Y_test.shape)   #20% go  for testing



feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
  #feature extraction for tags stop words is the english words like the had , has

X_train_features=feature_extraction.fit_transform(X_train)
X_test_features =feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test =Y_test.astype('int')
print(X_train)

print(X_train_features)  ##accuracy of model  0 means span and 1 means not spam

model = LogisticRegression() #train the model
model.fit(X_train_features,Y_train)

prediction_on_training_data =model.predict(X_train_features)
accuracy_on_training_data =accuracy_score(Y_train,prediction_on_training_data)
print('Accuracy on training data: ',accuracy_on_training_data)

prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(Y_test,prediction_on_test_data )
print('Accuracy on test data: ',accuracy_on_test_data)


input_your_mail =["Are you there"]  #input the mail message here
input_data_features = feature_extraction.transform(input_your_mail)

prediction= model.predict(input_data_features)

print(prediction)
if(prediction[0]==1):
    print('Ham')
else:
    print('Spam Mail')
