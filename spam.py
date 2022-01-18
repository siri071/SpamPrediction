#import all libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pickle

#data processing
rawMail = pd.read_csv("",encoding='latin-1') #load your directory in that empty string.
#replace the null values as null string
mailData = rawMail.where((pd.notnull(rawMail)),'')
#print(mailData)
#print(mailData.head()) # sample data

# label spam as 0 and ham as 1
mailData.loc[mailData['category']=='spam','category']=0
mailData.loc[mailData['category']=='ham','category']=1

#seperate the data
X = mailData['message']
Y = mailData['category']
'''print(x)
print(".........")
print(y)'''

# Train and Split
X_train , X_test , Y_train ,Y_test = train_test_split(X,Y,train_size=0.8,
test_size=0.2,random_state=3)

#transfer the text data into feature vectors that can be used as input to the SVM model
featureExt = TfidfVectorizer(min_df=1,stop_words='english',lowercase='True')
X_train_feature = featureExt.fit_transform(X_train)
X_test_feature = featureExt.transform(X_test)

#convert Y to integers
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

#training the model with SVM with training data
model = LinearSVC()
model.fit(X_train_feature,Y_train)

#prediction on training data
pred_train = model.predict(X_train_feature)
accuracy_train = accuracy_score(Y_train,pred_train)
print("Accuracy of training model: ",accuracy_train)

#prediction on testing data
pred_test = model.predict(X_test_feature)
accuracy_test = accuracy_score(Y_test,pred_test)
print("Accuracy of testing model: ",accuracy_test)

#Check the mail
input_mail = ["""Go until jurong point, crazy.. Available only in bugis n 
great world la e buffet... Cine there got amore wat..."""] #for dataset kindly visit kaggle.
input_mail_feature = featureExt.transform(input_mail)
prediction = model.predict(input_mail_feature)
print(prediction)

if prediction[0]==0:
    print("Spam mail")
else:
    print("Ham mail")

filename = 'spam_pred_model.pkl'
pickle.dump(model,open(filename,'wb'))