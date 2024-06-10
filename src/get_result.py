import joblib
import re
from process_string import lematize

class Classifier:
    def __init__(self, model_path='../data/model.pkl', vectorizer_path='../data/vectorizer.pkl'):
        self.clf = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def preprocess_sentence(self, sentence):
        processed_sentence = lematize(sentence)
        return processed_sentence

    def classify_sentence_with_proba(self, sentence):
        processed_sentence = self.preprocess_sentence(sentence)
        vec_sentence = self.vectorizer.transform([processed_sentence]).toarray()

        predicted_class = self.clf.predict(vec_sentence)[0]
        predicted_proba = self.clf.predict_proba(vec_sentence)[0]

        return predicted_class, predicted_proba

if __name__ == '__main__':
    classifier = Classifier()
    print(classifier.classify_sentence_with_proba("Михаела Филева се раздели с приятеля си"))

    predicted_class, predicted_proba = classifier.classify_sentence_with_proba("Ще има избори в България през 2025 година")
    print(f"Predicted class: {predicted_class}, Probability: {predicted_proba}")
