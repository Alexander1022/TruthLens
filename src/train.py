from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import joblib
from process_string import lematize

CSV_DIRECTORY = '../data/raw/'
DATAFILE = 'news.csv'
FILEPATH = CSV_DIRECTORY + DATAFILE

def check_class_distribution(filename):
    news = pd.read_csv(filename)
    class_distribution = news['label'].value_counts()
    print(class_distribution)

def train():
    news = pd.read_csv(FILEPATH)
    dataset = news.sample(frac=1)

    dataset['news_title'] = dataset['news_title'].apply(lambda x : lematize(x))

    X = dataset['news_title']
    y = dataset['label']

    train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.2, random_state=0)

    '''
    lowercase = False, защото вече сме премахнали стоп думите и превърнали текста в малки букви
    ngram_range = (1, 2) , защото искаме да ползваме униграми и биграми
    '''
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), lowercase=False)
    vec_train_data = vectorizer.fit_transform(train_data)
    vec_train_data = vec_train_data.toarray()
    vec_test_data = vectorizer.transform(test_data).toarray()

    '''
    Ще обучи предоставения класификатор чрез по-малка извадка от потока от дадени наблюдения, така че разпределението на класа, 
    видяно от класификатора, следва дадено желано разпределение
    '''
    oversampler = RandomUnderSampler(sampling_strategy=0.5)
    vec_train_data, train_label = oversampler.fit_resample(vec_train_data, train_label)
    training_data = pd.DataFrame(vec_train_data , columns=vectorizer.get_feature_names_out())
    testing_data = pd.DataFrame(vec_test_data , columns= vectorizer.get_feature_names_out())
    
    '''
    Задаване на повече итерации, за да се увеличи точността на модела
    Това ще работи добре с повече данни
    '''
    clf = LogisticRegression(max_iter=1000, verbose=2)
    clf.fit(training_data, train_label)
    y_pred = clf.predict(testing_data)

    print(classification_report(test_label , y_pred))
    
    '''
    Запазване на модела и векторизатора, за да може да се използва за предсказване на нови данни
    '''
    joblib.dump(clf, '../data/model.pkl')
    joblib.dump(vectorizer, '../data/vectorizer.pkl')

    return accuracy_score(test_label, y_pred)

check_class_distribution(FILEPATH)

accuracy = train()
print(accuracy)
