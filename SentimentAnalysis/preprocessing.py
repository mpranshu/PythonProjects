#importing all necessary packages
import nltk
#nltk.download('punkt')
from nltk.stem import PorterStemmer # root words
from nltk.tokenize import word_tokenize # sentence to words
from nltk.corpus import stopwords #to remove stop words
#nltk.download('stopwords')
import re
from bs4 import BeautifulSoup

#setting stopwords globally in preprocessing.py
stop_words = set(stopwords.words('english'))
stop_words.remove("not")
stop_words.add(".")

def text_Preprocessing(reviews):
    """ This will clean the text data, remove html tags, remove special characters and then tokenize the reviews to apply Stemmer on each word token."""
    
    pre_processed_reviews=[]
    
    for review in reviews:
        review= BeautifulSoup(review,'lxml').getText()    #remove html tags
        review=re.sub('\S*\d\S*','',review).strip() #remove any substring with numbers
        review=re.sub("n't"," not",review) #replace n't with not
        review=re.sub('[^A-Za-z]+',' ',review)        #remove special chars
        review=word_tokenize(str(review.lower())) #tokenize the reviews into word tokens
        # now we will split the review into words and then check if these words are in the stop words if so we will remove them, if not we will join
        print(review)
        review=' '.join(PorterStemmer().stem(word) for word in review if word not in stop_words)
        pre_processed_reviews.append(review.strip())
        print(pre_processed_reviews)
    return pre_processed_reviews

def word_Preprocessing(all_words):
    pre_processed_words=[]
    for word in all_words:
        if not word.isalpha():
            continue
        if word not in stop_words:
            pre_processed_words.append(word)
    return pre_processed_words
