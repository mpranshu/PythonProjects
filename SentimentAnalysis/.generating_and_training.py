import random
import nltk
from nltk.corpus import movie_reviews
import preprocessing
# using NLTK corpus movie reviews
from sklearn import model_selection
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
################################




################################
def generate():
    documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

    # shuffle the documents
    random.shuffle(documents) # to avoid bias data is in format +ve and -ve

    all_words = []
    for w in movie_reviews.words():
        all_words.append(w.lower())

    all_words = preprocessing.word_Preprocessing(all_words)
    all_words = nltk.FreqDist(all_words) #sort words from most common to least common
    word_features = list(all_words.keys())[:5000] #5000 most common words are feature
    
    # Now lets do it for all the documents
    featuresets = [(find_features(rev,word_features), category) for (rev, category) in documents]
    print(len(featuresets))
    training(featuresets)

def find_features(document,word_features):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def training(featuresets):
    # split the data into training and testing datasets
    training, testing = model_selection.train_test_split(featuresets, test_size = 0.25)#random_state=seed for reproducebility
    model = SklearnClassifier(SVC(kernel = 'rbf'))

    # train the model on the training data
    model.train(training)

    # and test on the testing dataset!
    accuracy = nltk.classify.accuracy(model, testing)*100
    print("SVC Accuracy: {}".format(accuracy))
generate()


