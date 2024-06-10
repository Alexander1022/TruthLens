import re
import spacy

def load_stopwords(filename):
    with open(filename, 'r', encoding="utf8") as file:
        stopwords = file.read().split('\n')
    return stopwords

# премахване на ненужни символи и стоп думи
def clean_data(data):
    data = data.lower()
    data = re.sub('[^а-яА-ЯёЁ]', ' ', data)
    token = data.split()
    stop_words = load_stopwords('../data/bulgarian_stopwords.txt')
    news = [word for word in token if not word in set(stop_words)]

    return ' '.join(news)

# основната форма (лема) на думите в текста
def lematize(data):
    data = clean_data(data)
    nlp = spacy.load('bg_news_lg')
    doc = nlp(data)
    news = [word.lemma_ for word in doc]

    return ' '.join(news)
