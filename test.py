from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=CountVectorizer()
corpus=["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]
# print(vectorizer.fit_transform(corpus))
# print(vectorizer.fit_transform(corpus).toarray())
# print( vectorizer.get_feature_names())
#
#
# transformer = TfidfTransformer()
# tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
# print(tfidf)
# print(tfidf.toarray())
vectorizer=CountVectorizer(max_features=6)
#vectorizer=TfidfVectorizer(max_df = 0.5,min_df=2)
print(vectorizer.fit_transform(corpus))
print(vectorizer.fit_transform(corpus).toarray())
print( vectorizer.get_feature_names())
