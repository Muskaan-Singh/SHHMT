import pandas as pd
FileName='/home/muskaan/scl/samsadhni_tenserflow'

data=pd.read_csv(FileName, sep=",",error_bad_lines=False)
#data.fillna('a', inplace=True)
numpyMatrix = data.as_matrix()
#print (numpyMatrix)
row = numpyMatrix[0]

abc=data['b'].values
print abc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vect = CountVectorizer()
d=vect.fit(abc)
vect.get_feature_names()
simple_train_dtm = vect.transform(abc)
simple_train_dtm
simple_train_dtm.toarray()

pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())

# check the type of the document-term matrix
type(simple_train_dtm)
print(simple_train_dtm)

simple_test=abc[0]

#simple_test_dtm = vect.transform(simple_test)
#simple_test_dtm.toarray()

#pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())
