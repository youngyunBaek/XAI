import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
              for x in newsgroups_train.target_names]

class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)

print(nb)

pred = nb.predict(test_vectors)
result = sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

print(result)