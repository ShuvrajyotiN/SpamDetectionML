import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import glob
import joblib

from pickle import TRUE

csv_files = glob.glob('model_training/*.csv')
columns_to_keep = ['subject', 'body', 'label']
df_list = [pd.read_csv(file, usecols = columns_to_keep) for file in csv_files]
df = pd.concat(df_list, ignore_index=True)

df['text'] = df['subject'].str.cat(df['body'], sep=' ', na_rep='')
df.drop(['subject','body'],axis = 1,inplace = True)
df = df[['text','label']]
df.head()

X = df['text']
Y = df['label']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state = 42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase = True)

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test =  Y_test.astype('int')

print(X_train_features)
print(X_test_features)


model = LogisticRegression()

model.fit(X_train_features,Y_train)

prediction_test = model.predict(X_test_features)
accuracy_test =  accuracy_score(Y_test,prediction_test)

print('Accuracy on testing data : ', accuracy_test)

confusion_matrix(Y_test, prediction_test)

joblib.dump(model, 'Spam_Detection_Model.pkl')
joblib.dump(feature_extraction, 'features.pkl')
