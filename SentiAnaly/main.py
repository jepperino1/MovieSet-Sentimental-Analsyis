import nltk
import string
from nltk import FreqDist, NaiveBayesClassifier, classify, DecisionTreeClassifier, MaxentClassifier, word_tokenize
from nltk.corpus import movie_reviews, stopwords

stopwords = stopwords.words('english')

positiveReviews = []
for fileName in movie_reviews.fileids('pos'): #Seperate all the positive reviews into a seperate list of reviews
    words = movie_reviews.words(fileName)
    positiveReviews.append(words)

print("A sample of positive reviews: " + str(positiveReviews[:10]))

negativeReviews = [] #Seperate all the negative reviews into a seperate list of reviews
for fileName in movie_reviews.fileids('neg'):
    words = movie_reviews.words(fileName)
    negativeReviews.append(words)

print("A sample of negative reviews: " + str(negativeReviews[:10]))

def bag_of_words(words):
    processedWords = []
    for word in words:
        word = word.lower()
        if word not in stopwords and word not in string.punctuation:
            processedWords.append(word)
    wordDictionary = dict([word, True] for word in processedWords) ##Create a dictionary so no words are repeated in the lexicon, Classifier also requires that words are followed by True
    return wordDictionary

positiveSet = []
for words in positiveReviews:
    positiveSet.append((bag_of_words(words), 'pos')) # Attach all of the words from positive reviews to a dict with a pos tag

negativeSet = []
for words in negativeReviews:
    negativeSet.append((bag_of_words(words), 'neg')) # Attach all of the words from negative reviews to a dict with a neg tag

from random import shuffle ##SHUFFLE THE REVIEWS
shuffle(positiveSet)
shuffle(negativeSet)

testingSet = positiveSet[:200] + negativeSet[:200]
trainingSet = positiveSet[200:] + negativeSet[200:]

classifierChoice = input("What classifier do you want? NB = Naive Bayes, DT = Decision Tree, MaxEnt = Maximum Entropy ") #Prompt user for their choice of classifier and then run their choice
if classifierChoice == "NB":
    print("Naive Bayes Selected, processing")
    classifier = NaiveBayesClassifier.train(trainingSet)
    print("Naive Bayes accuracy percent:", (classify.accuracy(classifier, testingSet)) * 100)
    print(classifier.show_most_informative_features(10))

if classifierChoice == "DT":
    print("Decision Tree Selected, processing")
    classifier = DecisionTreeClassifier.train(trainingSet, depth_cutoff = 3) #depth_cutoff = max depth of decision tree

    print("Decision Tree accuracy:", (classify.accuracy(classifier, testingSet)) * 100)

if classifierChoice == "MaxEnt":
    print("Maximum Entropy Selected, processing")
    classifier = MaxentClassifier.train(trainingSet, algorithm='gis', trace=3, max_iter=20) #trace = level of detail output, max_iter = maximum iterations that can be run
    print("Maximum Entropy accuracy:", (classify.accuracy(classifier, testingSet)) * 100)
    print(classifier.show_most_informative_features(10))


pos_custom_review = "Top quality walls here, they were steep and challenging but are definitely the most realistic around" #Run a sample positive review not movie related to see if it can correctly be classified
pos_custom_review_tokens = word_tokenize(pos_custom_review)
pos_custom_review_set = bag_of_words(pos_custom_review_tokens)
print (pos_custom_review + " \n Positive classification detected as: " + str(classifier.classify(pos_custom_review_set)))

neg_custom_review = "Rock walls here pretty disappointing, not many of them and they lack easier ones for beginners." #Run a sample negative review not movie related to see if it can correctly be classified
neg_custom_review_tokens = word_tokenize(neg_custom_review)
neg_custom_review_set = bag_of_words(neg_custom_review_tokens)
print (neg_custom_review + " \n Negative classification detected as: " + str(classifier.classify(neg_custom_review_set)))