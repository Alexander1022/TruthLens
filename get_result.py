import joblib
import re
from process_string import lematize
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_sentence(sentence):
    processed_sentence = lematize(sentence.lower())
    return processed_sentence

def classify_sentence(sentence, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    processed_sentence = preprocess_sentence(sentence)
    vec_sentence = vectorizer.transform([processed_sentence]).toarray()
    predicted_class = clf.predict(vec_sentence)

    return predicted_class[0]  

sentence = "На бреговете на река Дунав български учени откриха вълнуващи доказателства за съществуването на древна космическа цивилизация."
predicted_class = classify_sentence(sentence)
print("Predicted class:", predicted_class)
