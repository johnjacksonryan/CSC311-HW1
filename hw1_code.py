import random

import matplotlib.pyplot as plt
import numpy as np
from pylab import *
from math import *
import random
import numpy
import sympy as sympy
from matplotlib import pyplot as plt
from sklearn import tree
from sympy import *
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer


## CSC311 Homework 1 ##
## Question 2 ##

random.seed(1005024821)

def load_data():
    ## Load the data ##

    fake_news_file = open("clean_fake.txt", "r")
    real_news_file = open("clean_real.txt", "r")
    targets = []
    headlines = []
    while True:
        line = fake_news_file.readline()
        if not line:
            break
        headlines.append(line)
        targets.append("fake")
    while True:
        line = real_news_file.readline()
        if not line:
            break
        headlines.append(line)
        targets.append("real")

    ## Vectorize the headlines using count vectorizer ##

    vectorizer = CountVectorizer()
    vectorizer.fit(headlines)
    X = vectorizer.transform(headlines)

    ## Split the data into training, test and validation sets ##
    ## 15% test & 15% validation -> 490 test samples and 490 validation samples

    indexes = []
    for i in range(3266):
        indexes.append(i)
    random.shuffle(indexes)
    test_i = indexes[:490]
    validation_i = indexes[490:980]
    training_i = indexes[980:]

    training = X[training_i,]
    training_targets = []
    for i in training_i:
        training_targets.append(targets[i])
    test = X[test_i,]
    test_targets = []
    for i in test_i:
        test_targets.append(targets[i])
    validation = X[validation_i,]
    validation_targets = []
    for i in validation_i:
        validation_targets.append(targets[i])

    return training, training_targets, test, test_targets, validation, validation_targets, vectorizer.get_feature_names()


def select_model():
    data = load_data()
    training = data[0]
    training_targets = data[1]
    test = data[2]
    test_targets = data[3]
    validation = data[4]
    validation_targets = data[5]
    depths = [3, 5, 7, 10, 12]
    for d in depths:
        gini_classifier = DecisionTreeClassifier(criterion="gini", max_depth=d)
        gini_classifier.fit(training, training_targets)
        info_gain_classifier = DecisionTreeClassifier(criterion="entropy", max_depth=d)
        info_gain_classifier.fit(training, training_targets)
        if d == 12:
            fig1 = plt.figure(figsize=(25, 20))
            _ = tree.plot_tree(info_gain_classifier, max_depth=2, feature_names=data[6], class_names=training_targets,
                               filled=True)
            fig1.savefig("decision_tree_named.png")
            fig2 = plt.figure(figsize=(25, 20))
            _ = tree.plot_tree(info_gain_classifier, max_depth=2, class_names=training_targets,
                               filled=True)
            fig2.savefig("decision_tree_indices.png")
        val_pred_gini = gini_classifier.predict(validation)
        val_pred_info_gain = info_gain_classifier.predict(validation)
        gini_val_error = 0
        info_gain_val_error = 0
        for i in range(490):
            if val_pred_gini[i] != validation_targets[i]:
                gini_val_error += 1
            if val_pred_info_gain[i] != validation_targets[i]:
                info_gain_val_error += 1
        print("max_depth: " + str(d))
        print("Gini Error: " + str(gini_val_error/490))
        print("Info Gain Error: " + str(info_gain_val_error/490))

#select_model()


def compute_information_gain(feature, split):
    data = load_data()
    training = data[0]
    training_targets = data[1]
    total = len(training_targets)
    num_fake = 0
    num_real = 0
    for target in training_targets:
        if target == "fake":
            num_fake += 1
        else:
            num_real += 1
    p_fake = num_fake/total
    p_real = num_real/total
    target_entropy = -(p_fake*log(p_fake, 2) + p_real*log(p_real, 2))
    num_fake_below_split = 0
    num_fake_above_split = 0
    num_real_below_split = 0
    num_real_above_split = 0
    for index in range(total):
        if training[index, feature] < split:
            if training_targets[index] == "fake":
                num_fake_below_split += 1
            else:
                num_real_below_split += 1
        else:
            if training_targets[index] == "fake":
                num_fake_above_split += 1
            else:
                num_real_above_split += 1
    p_fake_and_below = num_fake_below_split/total
    p_fake_and_above = num_fake_above_split/total
    p_real_and_below = num_real_below_split/total
    p_real_and_above = num_real_above_split/total
    p_fake_given_below = num_fake_below_split/(num_real_below_split + num_fake_below_split)
    p_fake_given_above = num_fake_above_split/(num_real_above_split + num_fake_above_split)
    p_real_given_below = num_real_below_split/(num_real_below_split + num_fake_below_split)
    p_real_given_above = num_real_above_split/(num_real_above_split + num_fake_above_split)
    target_cond_entropy = -(p_fake_and_below*log(p_fake_given_below, 2) + p_fake_and_above*log(p_fake_given_above, 2) +
                            p_real_and_below*log(p_real_given_below, 2) + p_real_and_above*log(p_real_given_above, 2))
    information_gain = target_entropy - target_cond_entropy
    return information_gain

## Information Gain ##
print("Information Gain on Splits")
## Note the indices of the features the, hillary, and trumps are 5143, 2405, and 5324 respectively
## taken from the unlabelled verson of the descision tree display
print("the: " + str(compute_information_gain(5143, 0.5)))
print("hillary: " + str(compute_information_gain(2405, 0.5)))
print("trumps: " + str(compute_information_gain(5324, 0.5)))

