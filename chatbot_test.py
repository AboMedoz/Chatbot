import json
import pickle

import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
with open("intents.json", "r") as file:
    intents = json.load(file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_model.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [1 if w in sentence_words else 0 for w in words]
    return np.array(bag).reshape(1, -1)


def predict_intent(sentence):
    bow_vector = bow(sentence, words)
    prediction = model.predict(bow_vector)[0]
    max_index = np.argmax(prediction)  # Get the class with highest probability
    intent = classes[max_index]
    return intent, prediction[max_index]


def get_response(intent):
    for i in intents["intents"]:
        if i["tag"] == intent:
            return np.random.choice(i["responses"])


def chatbot():
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("Chatbot: Goodbye!")
            break

        intent, confidence = predict_intent(user_input)
        if confidence > 0.5:
            response = get_response(intent)
        else:
            response = "I'm sorry, I didn't understand that."

        print(f"Chatbot: {response}")


# Run chatbot
if __name__ == "__main__":
    chatbot()
