#!/usr/bin/python3
# Script that trains a classifier for target token detection
# Filename: target_token_classifier.py
# Author: A.J. Schelhaas
# Date: 20-05-2020

import spacy
import random


def read_files(file):
	with open(file, "r", encoding='utf-8') as f:
		data = f.readlines()

	return data


def merge_entities(tokens):
	entity_list = []
	tokens_len = len(tokens)
	index_used = []
	for index, token in enumerate(tokens):
		if token.ent_type_ != "" and index not in index_used:
			# find entity
			entity = [token]
			index_next = index+1
			while index_next is not None and index_next < tokens_len and index_next not in index_used:
				next_token = tokens[index_next]
				if next_token.ent_type_ != "":
					entity.append(next_token)
					index_used.append(index_next)
					index_next += 1
				else:
					index_next = None

			# create entities tuples
			entity_string_list = list()
			for element in entity:
				entity_string_list.append(element.text)
			entity_string = " ".join(entity_string_list)
			entity_tuple = (entity_string, token.ent_type_)
			entity_list.append(entity_tuple)

	return entity_list


def get_target_token(nlp, string, label):
	target_token = ""

	ner_dic = dict()
	ner_dic["INDIVIDUAL"] = ["PERSON"]
	ner_dic["GROUP"] = ["NORP", "GPE", "LANGUAGE"]
	ner_dic["OTHER"] = ["FAC", "ORG", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW"]

	if label != "NA":
		tokens = nlp(string)
		entity_list = merge_entities(tokens)
		ner_get = ner_dic[label]
		for entity_string, label in entity_list:
			if label in ner_get:
				target_token = entity_string
		# assign first if label not found
		if target_token == "" and len(entity_list) > 0:
			target_token = entity_list[0][0]

	return target_token


def main(dev_mode=False, file="Dutch_Abusive_Language_Corpus_Train.tsv"):
	nlp = spacy.load("nl_core_news_sm")

	predicted_list = []

	data = read_files(file)
	for line in data:
		line_split = line.strip().split("\t")
		tweet_text = line_split[1]
		tweet_label = line_split[-1]

		target_token = get_target_token(nlp, tweet_text, tweet_label)
		if tweet_label != "NA":
			predicted_list.append((target_token, tweet_text))

	for element in predicted_list:
		target_token = element[0]
		text = element[1]
		string = "{:32} {}".format(target_token, text)
		print(string)


if __name__ == "__main__":
	main(True)