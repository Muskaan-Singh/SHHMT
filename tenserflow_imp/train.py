import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

path = '/home/mandeep/PycharmProjects/samsadhni/data.csv'
sms = pd.read_table(path, sep=',', error_bad_lines=False)
sms.shape
sms.head(10)
# examine the class distribution
# print sms.a.value_counts()
sms['label_num'] = sms.a.map({'ham': 0, 'spam': 1})
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

print type(X_test)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)
# instantiate the vectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
# vect.fit(X_train)
vect = CountVectorizer(
    stop_words='english',
    ngram_range=(1, 1),  # ngram_range=(1, 1) is the default
    dtype='double',
)
# print type(X_train)

data = vect.fit_transform(X_train)

X_train_dtm = vect.transform(X_train)
# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix

# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
y_test_dtm = vect.fit_transform(y_test)

print X_test_dtm.shape
print y_test_dtm.shape





# import and instantiate a Multinomial Naive Bayes model
# import and instantiate a logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
# train the model using X_train_dtm
logreg.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_class = logreg.predict(X_test_dtm)
# print (y_pred_class)
# calculate predicted probabilities y_test_dtmfor X_test_dtm (well calibrated)
y_pred_prob = logreg.predict_proba(X_test_dtm)[:, 1]
print(y_pred_prob)
print (y_test)
# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class)*100)
