#Define Machine Learning Algorithm HERE

from main import preprocess, Vocab
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

vocab = Vocab()

X_aws = preprocess('aws_title.txt', vocab)
X_azure = preprocess('azure_title.txt', vocab)
X_gcp = preprocess('gcp_title.txt', vocab)

X = X_aws + X_azure + X_gcp
Y = [0 for _ in range(len(X_aws))] + [1 for _ in range(len(X_azure))] + [2 for _ in range(len(X_gcp))]

X_aws, X_azure, X_gcp = 0, 0, 0

cv = CountVectorizer(max_df=0.85, max_features=5000)
X = cv.fit_transform(X)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X = tfidf_transformer.fit_transform(X)

X, Y = shuffle(X, Y)

X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)

lr = LogisticRegression()
lr.fit(X, Y)
print(lr.score(X, Y))
print(lr.score(X_test, Y_test))

knn = KNeighborsClassifier()
knn.fit(X, Y)
print(knn.score(X, Y))
print(knn.score(X_test, Y_test))

nb = GaussianNB()
nb.fit(X, Y)
print(nb.score(X, Y))
print(nb.score(X_test, Y_test))

# svm = SVC()
# svm.fit(X, Y)
# print(svm.score(X_test, Y_test))

# dt = DecisionTreeClassifier()
# dt.fit(X, Y)
# print(dt.score(X, Y))
# print(dt.score(X_test, Y_test))
#
# rf = RandomForestClassifier()
# rf.fit(X, Y)
# print(rf.score(X, Y))
# print(rf.score(X_test, Y_test))
