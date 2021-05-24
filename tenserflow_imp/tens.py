import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


path = '/home/mandeep/PycharmProjects/samsadhni/data.csv'
sms = pd.read_table(path,sep=',',error_bad_lines=False)
sms.shape
sms.head(10)
# examine the class distribution
# print sms.a.value_counts()
sms['label_num'] = sms.a.map({'ham':0, 'spam':1})
sms.head(10)

# how to define X and y (from the iris data) for use with a MODEL
X = sms.get('b','c')
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
print type(X_train)
X_train_dtm = vect.fit_transform(X_train)
print X_train_dtm
Y_train_dtm =vect.fit_transform(y_train)
