#!/usr/bin/python3
# Script that trains a classifier for target type detection
# Filename: target_type_classifier.py
# Author: A.J. Schelhaas
# Date: 18-05-2020

import nltk.classify
import collections
from nltk import ngrams
from sklearn.metrics import confusion_matrix
from nltk.metrics import precision, recall
from nltk.tokenize import word_tokenize
from featx import bag_of_words, high_information_words
from random import shuffle
from sklearn.svm import SVC
from nltk.corpus import stopwords
from distutils import util
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
import spacy


# Reads files for each category
def read_files(pre_processing=False, ner_tags=False, add_ngrams=False, train_data=True):
	""" Reads the files for each category (asshole and not_asshole). """
	feats = list()
	print("\n##### Reading files...")
	stop_words = set(stopwords.words('dutch'))

	# load in ner tagger
	ner_list = list()
	if ner_tags:
		nlp = spacy.load("nl_core_news_sm")

	# ngrams list
	ngrams_list = list()

	# load train data
	if train_data:
		file = "Dutch_Abusive_Language_Corpus_Train.tsv"
	# load test data
	else:
		file = "Dutch_Abusive_Language_Corpus_Test.tsv"

	with open(file, "r", encoding="utf-8") as f:
		data = f.readlines()

	print("Loaded", str(len(data)), "tweets")

	for line in data:
		line_split = line.split("\t")
		tweet_text = line_split[1].strip().lower()
		tweet_label = line_split[-1].strip()
		tokens = word_tokenize(tweet_text)

		print(tweet_label, line_split[-2].strip(), tweet_text)

		if tweet_label != "NA":
			# get ner tags for text
			ner_set = set()
			if ner_tags:
				parsed_string = nlp(tweet_text)
				for token in parsed_string:
					if token.ent_type_ != "":
						ner_set.add(token.ent_type_)
				ner_list.append(ner_set)

			# Perform pre-processing to tweet
			if pre_processing:
				# lower and strip tokens
				new_tokens = list()
				for token in tokens:
					new_tokens.append(token.lower().strip())
				tokens = new_tokens

				# remove stopwords DEPRECATED
				'''
				new_tokens = list()
				for token in tokens:
					if token not in stop_words:
						new_tokens.append(token)
				tokens = new_tokens
				'''

			# get n-grams for text
			ngrams_set = set()
			if add_ngrams:
				ngrams_object = ngrams(tokens, 2)
				for grams in ngrams_object:
					ngrams_set.add(" ".join(grams))
				ngrams_list.append(ngrams_set)

			# Turn tokens into a bag of words
			bag = bag_of_words(tokens)
			feats.append((bag, tweet_label))

	print("Using", str(len(feats)), "tweets")

	return feats, ner_list, ngrams_list


# Function to split the dataset for n fold cross validation
def split_folds(feats, folds=10):
	""" Splits the datasets into n amount of folds, default is set to 10 folds. """
	shuffle(feats)  # randomise dataset before splitting into train and test
	subset_size = int(len(feats) / folds)
	# divide feats into N cross fold sections
	nfold_feats = []
	for n in range(folds):
		test_feats = feats[n * subset_size:][:subset_size]
		train_feats = feats[:n * subset_size] + feats[(n + 1) * subset_size:]
		nfold_feats.append((train_feats, test_feats))

	print("\n##### Splitting datasets...")
	return nfold_feats


# Trains a classifier
def train(train_feats, svm=True, c=3):
	""" Trains a classifier with a given feature set, by default it uses an SVM classifier, for which the C value can
	be adjusted. The function can also be set to train a Naive Bayes classifier. """
	if svm:
		classifier = nltk.classify.SklearnClassifier(SVC(C=c, kernel='linear')).train(train_feats)
	else:
		classifier = nltk.classify.NaiveBayesClassifier.train(train_feats)

	return classifier


# Calculates F-score
def calculate_f(precisions, recalls):
	""" Calculates the F-score for given precision and recall scores. """
	f_measures = {}
	keys = precisions.keys()
	for key in keys:
		try:
			f_measures[key] = round((2 * precisions[key] * recalls[key]) / (precisions[key] + recalls[key]), 6)
		except (ZeroDivisionError, TypeError):
			f_measures[key] = "NA"
	return f_measures


# Returns the precision and recall for a given classifier and a featureset
def precision_recall(classifier, testfeats):
	""" Calculates and returns the precision and recall for a given classifier """
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)

	precisions = {}
	recalls = {}

	for label in classifier.labels():
		precisions[label] = precision(refsets[label], testsets[label])
		recalls[label] = recall(refsets[label], testsets[label])

	return precisions, recalls


# Show most informative features
def analysis(classifier, svm):
	""" Shows the features that were most informative for a given classifier. """
	print("\n##### Most informative features...")
	if svm:
		print("Most informative features not available for Support Vector Machines")
	else:
		classifier.show_most_informative_features()


# Prints Accuracy, Precision and Recall
def evaluation(classifier, test_feats, categories, print_results=True):
	""" Gives the results and the evaluation of the results for a given classifier in a table. """
	accuracy = nltk.classify.accuracy(classifier, test_feats)
	precisions, recalls = precision_recall(classifier, test_feats)
	f_measures = calculate_f(precisions, recalls)
	if print_results:
		print("\n##### Evaluation...")
		print("  Accuracy: %f" % accuracy)

		print(" |-----------|-----------|-----------|-----------|")
		print(" |%-11s|%-11s|%-11s|%-11s|" % ("category", "precision", "recall", "F-measure"))
		print(" |-----------|-----------|-----------|-----------|")
		for category in categories:
			if precisions[category] is None:
				print(" |%-11s|%-11s|%-11s|%-11s|" % (category, "NA", "NA", "NA"))
			else:
				try:
					print(" |%-11s|%-11f|%-11f|%-11s|" % (category, precisions[category], recalls[category], f_measures[category]))
				except TypeError:
					pass
		print(" |-----------|-----------|-----------|-----------|")

	return accuracy


def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("\nNormalized confusion matrix")
	else:
		print('\nConfusion matrix, without normalization')

		print(cm)
		print()

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.gcf().subplots_adjust(bottom=0.25)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig('confusion_matrix.png')


# Calculates and prints confusion matrix for the classifier
def confusion_matrix_function(classifier, test_feats, print_results=True):
	""" Calculates and prints confusion matrix for the classifier """
	results = list()
	labels = list()
	for i, (feats, label) in enumerate(test_feats):
		observed = classifier.classify(feats)
		results.append(observed)
		labels.append(label)

	cm = confusion_matrix(labels, results)
	plot_confusion_matrix(np.array(cm), classes=['INDIVIDUAL', 'GROUP', 'OTHER'],title="")

	return cm


# Obtain the high information words
def high_information(feats, categories):
	""" Returns a list with high information words based on a feature set. """
	print("\n##### Obtaining high information words...")

	# 1. convert the formatting of our features to that required by high_information_words
	from collections import defaultdict
	words = defaultdict(list)
	all_words = list()
	for category in categories:
		words[category] = list()

	for feat in feats:
		category = feat[1]
		bag = feat[0]
		for w in bag.keys():
			words[category].append(w)
			all_words.append(w)

	labelled_words = [(category, words[category]) for category in categories]

	# 2. calculate high information words
	high_info_words = set(high_information_words(labelled_words, min_score=2))

	print("  Number of words in the data: %i" % len(all_words))
	print("  Number of distinct words in the data: %i" % len(set(all_words)))
	print("  Number of distinct 'high-information' words in the data: %i" % len(high_info_words))

	return high_info_words


# Get high info words for a single tweet
def get_high_information_features_tweet(post, label, high_info_words):
	""" Returns tuple with features in combination with label """
	filtered_features = {}
	for feature in post.keys():
		token = feature.lower()
		if token in high_info_words:
			filtered_features[token] = True

	info_tuple = (filtered_features, label)

	return info_tuple


# Get high information features for all tweets
def get_high_information_features(feats, high_info_words):
	""" Returns features for all posts """
	high_information_feats_list = []
	for post, label in feats:
		high_information_feats = get_high_information_features_tweet(post, label, high_info_words)
		high_information_feats_list.append(high_information_feats)

	return high_information_feats_list


# Add the ner tags to a feature set
def add_ner_tags_to_features(feats, ner_list):
	""" Add the ner tags to a feature set """
	index = 0
	new_feats = list()
	for features, label in feats:
		ner_tags = ner_list[index]
		for tag in ner_tags:
			features[tag] = True
		new_feats.append((features, label))
		index += 1

	return new_feats


# Add the ngrams to a feature set
def add_ngrams_to_features(feats, ngrams_list):
	""" Add the ngrams to a feature set """
	index = 0
	new_feats = list()
	for features, label in feats:
		ngrams_set = ngrams_list[index]
		for ngram in ngrams_set:
			features[ngram] = True
		new_feats.append((features, label))
		index += 1

	return new_feats


# main
def main(svm=False, pre_processing=False, ner_tags=False, high_info=False, add_ngrams=False, train_mode=False):
	categories = ["INDIVIDUAL", "GROUP", "OTHER"] #, "NA"]

	# Load categories from dataset
	train_feats, ner_list, ngrams_list = read_files(pre_processing, ner_tags, add_ngrams, True)

	# Retrieves most informative words
	high_info_words = set()
	if high_info:
		high_info_words = high_information(train_feats, categories)
		train_feats = get_high_information_features(train_feats, high_info_words)

	# Add ner tags to features
	if ner_tags:
		train_feats = add_ner_tags_to_features(train_feats, ner_list)

	# Add ngrams to features
	if add_ngrams:
		train_feats = add_ngrams_to_features(train_feats, ngrams_list)

	# Split data into 10 folds
	if train_mode:
		split_feats = split_folds(train_feats)

	# Load and prepare test data if not training
	else:
		test_feats, ner_list, ngrams_list = read_files(pre_processing, ner_tags, add_ngrams, False)

		# Retrieves most informative words
		if high_info:
			test_feats = get_high_information_features(test_feats, high_info_words)

		# Add ner tags to features
		if ner_tags:
			test_feats = add_ner_tags_to_features(test_feats, ner_list)

		# Add ngrams to features
		if add_ngrams:
			test_feats = add_ngrams_to_features(test_feats, ngrams_list)

		split_feats = [(train_feats, test_feats)]

	# Perform classification
	accuracies = []
	for train_feats, test_feats in split_feats:
		classifier = train(train_feats, svm, 2)
		accuracy = evaluation(classifier, test_feats, categories, True)
		accuracies.append(accuracy)
		analysis(classifier, svm)
		confusion_matrix_function(classifier, test_feats)

	# Show accuracies
	print("\n##### Accuracy...")
	for acc in accuracies:
		print(acc)

	# show mean accuracy
	if train_mode:
		mean_accuracy = sum(accuracies) / len(accuracies)
		print(mean_accuracy)


if __name__ == "__main__":
	classifier_type = sys.argv[1]  # String to indicate which classifier to use: 'svm' or 'nb'
	svm_setting = True
	if classifier_type == 'nb':
		svm_setting = False

	pre_processing = util.strtobool(sys.argv[2])  # Boolean value to indicate whether to use pre-processing
	ner_tags = util.strtobool(sys.argv[3])  # Boolean to indicate whether to use ner tags
	high_info = util.strtobool(sys.argv[4])  # Boolean to indicate whether to use high info features
	add_ngrams = util.strtobool(sys.argv[5])  # Boolean to indicate whether to use ner tags
	train_mode = util.strtobool(sys.argv[6])  # Boolean to indicate whether to run in development or test mode

	main(svm_setting, pre_processing, ner_tags, high_info, add_ngrams, train_mode)
