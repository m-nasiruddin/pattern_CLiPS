#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from pattern.vector import words, count, stem, PORTER, LEMMA, chngrams, Document, Vector, distance, Model, TFIDF,\
    HIERARCHICAL, Cluster, NB, kfoldcv, KNN, EUCLIDEAN, TF, SVM, RADIAL, gridsearch, GA
from pattern.en import parse, Sentence, parsetree, lexicon
from pattern.db import csv
from random import choice


# word count
freq_dic = {}
with open('data/input/corpus.txt', 'r') as fp:
    words_list = words(fp.read(), filter=lambda w: w.strip("'").isalnum(), punctuation='.,;:!?()[]{}`''\"@#$^&*+-|=~_')
    # returns a list of words by splitting the string on spaces.
    freq_dic = count(  # takes a list of words and returns a dictionary of (word, count)-items.
        words=words_list,
        top=None,  # Filter words not in the top most frequent (int).
        threshold=0,  # Filter words whose count <= threshold.
        stemmer=None,  # PORTER | LEMMA | function | None
        exclude=[],  # Filter words in the exclude list.
        stopwords=False,  # Include stop words?
        language='en')  # en, es, de, fr, it, nl
for k, v in freq_dic.iteritems():
    print k, v
# stop words and stemming
print stem('spies', stemmer=PORTER)
print stem('spies', stemmer=LEMMA)
s = 'The black cat was spying on the white cat.'
print count(words(s), stemmer=PORTER)
print count(words(s), stemmer=LEMMA)
s = 'The black cat was spying on the white cat.'
s = Sentence(parse(s))
print count(s, stemmer=LEMMA)
# character n-grams
print chngrams('The cat sat on the mat.'.lower(), n=3)
# document
text = "The shuttle Discovery, already delayed three times by technical problems and bad weather, was grounded again" \
    "Friday, this time by a potentially dangerous gaseous hydrogen leak in a vent line attached to the shipʼs" \
    "external tank. The Discovery was initially scheduled to make its 39th and final flight last Monday, bearing" \
    "fresh supplies and an intelligent robot for the International Space Station. But complications delayed the" \
    "flight from Monday to Friday, when the hydrogen leak led NASA to conclude that the shuttle would not be ready" \
    "to launch before its flight window closed this Monday."
doc = Document(text, threshold=1)
print doc.keywords(top=6)
document = Document(text,
                    filter=lambda w: w.lstrip("'").isalnum(),
                    punctuation='.,;:!?()[]{}\'`"@#$*+-|=~_',
                    top=None,  # Filter words not in the top most frequent.
                    threshold=0,  # Filter words whose count falls below threshold.
                    exclude=[],  # Filter words in the exclude list.
                    stemmer=None,  # STEMMER | LEMMA | function | None.
                    stopwords=False,  # Include stop words?
                    name=None,
                    type=None,
                    language=None,
                    description=None)
print document.id  # Unique number (read-only).
print document.name  # Unique name, or None, used in Model.document().
print document.type  # Document type, used with classifiers.
print document.language  # Document language (e.g., 'en').
print document.description  # Document info.
print document.model  # The parent Model, or None.
print document.features  # List of words from Document.words.keys().
print document.words  # Dictionary of (word, count)-items (read-only).
print document.wordcount  # Total word count.
print document.vector  # Cached Vector (read-only dict).
print document.tf('conclude')  # returns the frequency of a word as a number between 0.0-1.0.
print document.tfidf('conclude')  # returns the word's relevancy as tf-idf. Note: simply yields tf if model is None.
print document.keywords(top=10, normalized=True)  # returns a sorted list of (weight, word)-tuples. With normalized=True
# the weights will be between 0.0-1.0 (their sum is 1.0).
print document.copy()
# document vector
v1 = Vector({"curiosity": 1, "kill": 1, "cat": 1})
v2 = Vector({"curiosity": 1, "explore": 1, "mars": 1})
print 1 - distance(v1, v2)
# model
d1 = Document('A tiger is a big yellow cat with stripes.', type='tiger')
d2 = Document('A lion is a big yellow cat with manes.', type='lion',)
d3 = Document('An elephant is a big grey animal with a slurf.', type='elephant')
print d1.vector
m = Model(documents=[d1, d2, d3], weight=TFIDF)
print d1.vector
print m.similarity(d1, d2)  # tiger vs. lion
print m.similarity(d1, d3)  # tiger vs. elephant
# lsa concept space
d1 = Document('The cat purrs.', name='cat1')
d2 = Document('Curiosity killed the cat.', name='cat2')
d3 = Document('The dog wags his tail.', name='dog1')
d4 = Document('The dog is happy.', name='dog2')
m = Model([d1, d2, d3, d4])
m.reduce(2)
for d in m.documents:
    print
    print d.name
    for concept, w1 in m.lsa.vectors[d.id].items():
        for feature, w2 in m.lsa.concepts[concept].items():
            if w1 != 0 and w2 != 0:
                print (feature, w1 * w2)
# clustering
d1 = Document('Cats are independent pets.', name='cat')
d2 = Document('Dogs are trustworthy pets.', name='dog')
d3 = Document('Boxes are made of cardboard.', name='box')
m = Model((d1, d2, d3))
print m.cluster(method=HIERARCHICAL, k=2)
# hierarchical clustering
cluster = Cluster((1, Cluster((2, Cluster((3, 4))))))
print cluster.depth
print cluster.flatten(1)
# training a classifier
nb = NB()
for review, rating in csv('data/input/reviews.csv'):
    v = Document(review, type=int(rating), stopwords=True)
    nb.train(v)
print nb.classes
print nb.classify(Document('A good movie!'))
# testing a classifier
data = csv('data/input/reviews.csv')
data = [(review, int(rating)) for review, rating in data]
data = [Document(review, type=rating, stopwords=True) for review, rating in data]
nb = NB(train=data[:500])
accuracy, precision, recall, f1 = nb.test(data[500:])
print accuracy
# binary classification
data = csv('data/input/reviews.csv')
data = [(review, int(rating) >= 3) for review, rating in data]
data = [Document(review, type=rating, stopwords=True) for review, rating in data]
nb = NB(train=data[:500])
accuracy, precision, recall, f1 = nb.test(data[500:])
print accuracy, precision, recall, f1
# confusion matrix
print nb.distribution
print nb.confusion_matrix(data[500:])
print nb.confusion_matrix(data[500:])(True)  # (TP, TN, FP, FN)
# precision and recall
print nb.test(data[500:], target=True)
print nb.test(data[500:], target=False)
print nb.test(data[500:])
# k-fold cross validation
data = csv('data/input/reviews.csv')
data = [(review, int(rating) >= 3) for review, rating in data]
data = [Document(review, type=rating, stopwords=True) for review, rating in data]
print kfoldcv(NB, data, folds=10)
print kfoldcv(KNN, data, folds=10, k=3, distance=EUCLIDEAN)
# feature selection


def v(review1):
    v3 = parsetree(review1, lemmata=True)[0]
    v4 = [w.lemma for w in v3 if w.tag.startswith(('JJ', 'NN', 'VB', '!'))]
    v5 = count(v4)
    return v5


data = csv('data/input/reviews.csv')
data = [(v(review), int(rating) >= 3) for review, rating in data]
print kfoldcv(NB, data)
data = csv('data/input/reviews.csv')
data = [(review, int(rating) >= 3) for review, rating in data]
data = [Document(review, type=rating, stopwords=True) for review, rating in data]
model = Model(documents=data, weight=TF)
model = model.filter(features=model.feature_selection(top=1000))
print kfoldcv(NB, model)
# gridsearch
data = csv('data/input/reviews.csv')
data = [(count(review), int(rating) >= 3) for review, rating in data]
for (A, P, R, F, o), p in gridsearch(SVM, data, kernel=[RADIAL], gamma=[0.1, 1, 10]):
    print (A, P, R, F, o), p
print kfoldcv(SVM, data, folds=10)
# genetic algorithm


def chseq(length=4, chars='abcdefghijklmnopqrstuvwxyz'):
    # returns a string of random characters.
    return ''.join(choice(chars) for i1 in range(length))


class Jabberwocky(GA):
    def fitness(self, w):
        return sum(0.2 for ch in chngrams(w, 4) if ch in lexicon) + sum(0.1 for ch in chngrams(w, 3) if ch in lexicon)

    def combine(self, w3, w4):
        return w3[:len(w3)/2] + w4[len(w4)/2:]  # cut-and-splice

    def mutate(self, w):
        return w.replace(choice(w), chseq(1), 1)


# Start with 10 strings, each 8 random characters.
candidates = [''.join(chseq(8)) for i in range(10)]
ga = Jabberwocky(candidates)
i = 0
while ga.avg < 1.0 and i < 1000:
    ga.update(top=0.5, mutation=0.3)
    i += 1
print ga.population
print ga.generation
print ga.avg
