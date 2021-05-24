from keras.models import Sequential, model_from_json
from keras.layers import Dense
import numpy as np
import h5py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from tens import Y_train_dtm

encoding=request.get("encoding");
sentences=request.get("text");
splitter=request.get("splitter");
out_encoding=request.get("out_encoding");
parse=request.get("parse");
text_type=request.get("text_type");

path = '/home/muskaan/scl/samsadhni_tenserflow/data.csv'
sms = pd.read_table(path,sep=',',error_bad_lines=False)
sms.shape
sms.head(10)
# examine the class distribution
# print sms.a.value_counts()
sms['label_num'] =( sms.a.map({'ham':0, 'spam':1}),sentences)
sms.head(10)

# how to define X and y (from the iris data) for use with a MODEL
X = sms.get('b')
y = sms.get('c','d')
# print(X.shape)
# print(y.shape)
# store the feature matrix (X) and response vector (y
# check the shapes of X and y
# print(X.shape)
# print(y.shape)

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
#vect.fit(X_train)
vect = CountVectorizer(
    stop_words='english',
    ngram_range=(1, 1),  #ngram_range=(1, 1) is the default
    dtype='double',
)

X_train_dtm = vect.fit_transform(X_train)
print (X_train_dtm)
Y_train_dtm =vect.fit_transform(y_train)
print (Y_train_dtm)


# fix random seed for reproducibility
#numpy.random.seed(7)
# load pima indians dataset
#dataset = numpy.loadtxt("/home/mandeep/PycharmProjects/samsadhni/data.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = X_train_dtm[:,0:723]
Y = Y_train_dtm[:,723]

X= X.toarray(order=None, out=None)
Y= Y.toarray(order=None, out=None)

# create model
model = Sequential()
model.add(Dense(12, input_dim=723, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=20, batch_size=50,  verbose=2)
# calculate predictions
predictions = model.predict(X)
# round predictions
rounded = [x*10000 for x in predictions]
print(rounded)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create modelX[0]
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)

