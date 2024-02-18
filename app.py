import streamlit as st
import nltk
import random
import json
import numpy as np
import pandas as pd

# List of resources to check and download if necessary
resources = ['punkt', 'stopwords', 'wordnet']

for resource in resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource)
    else:
        print(f"NLTK resource {resource} is already downloaded.")

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

# Function to preprocess user input
def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if not w in stopwords.words('english')]
    return filtered_words

# Function to extract features from user input
def extractFeatures(text):
    words = preprocess(text)
    lmtzr = WordNetLemmatizer()
    result = [lmtzr.lemmatize(word) for word in words]
    return result

def word_feats(words):
    return dict([(word, True) for word in words])

# load data
training_data = np.load('training_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy' , allow_pickle=True)

def extract_feature_from_doc(data):
    result = []
    # Corpus - Collection of text
    corpus = []
    # The responses of the chat bot
    answers = {}
    for (text,category,answer) in data:

        features = extractFeatures(text)

        corpus.append(features)
        result.append((word_feats(features), category))
        answers[category] = answer
    combined_corpus = [word for sublist in corpus for word in sublist]
    return (result, combined_corpus, answers)

def get_content(filename):
    with open(filename, 'r') as content_file:
        data = json.load(content_file)
        all_data = []
        for intent in data['intents']:
            for pattern, response in zip(intent['patterns'], intent['responses']):
                all_data.append([pattern, intent['tag'], response])
    return all_data

filename = 'data.json'
data = get_content(filename)

features_data, corpus, answers = extract_feature_from_doc(data)

def train_using_naive_bayes(training_data, test_data):
    classifier = nltk.NaiveBayesClassifier.train(training_data)
    classifier_name = type(classifier).__name__
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

classifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_naive_bayes(training_data, test_data)

def chatbot_response(input_text):
    category = classifier.classify(word_feats(extractFeatures(input_text)))
    return answers[category]

def main():
    st.title("Medical Chatbot")

    user_input = st.text_input("You: ")

    if st.button("Send"):
        if user_input:
            st.write("User: " + user_input)
            bot_response = chatbot_response(user_input)
            st.write("Chatbot: " + bot_response)

if __name__ == "__main__":
    main()
