from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from tens import Y_train_dtm

path = '/home/mandeep/PycharmProjects/samsadhni/data.csv'
sms = pd.read_table(path,sep=',',error_bad_lines=False)
sms.shape
sms.head(10)
# examine the class distribution
# print sms.a.value_counts()
sms['label_num'] = sms.a.map({'ham':0, 'spam':1})
sms.head(10)

# how to define X and y (from the iris data) for use with a MODEL
X = sms.get('b')
y = sms.get('c')
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
print ("sdg")
print (Y_train_dtm)




# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=X_train_dtm[30:],
    target_dtype=np.int,
    features_dtype=np.unicode)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=X_train_dtm[:30],
    target_dtype=np.int,
    features_dtype=np.unicode)


  # Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=2)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/iris_model")
  # Define the training inputs
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.
classifier.fit(input_fn=get_train_inputs, steps=1000)

  # Define the test inputs
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

# Classify two new flower samples.
def new_samples():
 return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

predictions = list(classifier.predict(input_fn=new_samples))

print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))


