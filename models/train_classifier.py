import sys
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import GridSearchCV
from joblib import dump
import numpy as np
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath):
    """Load database file and return input e target to train
    Args:
        database_filepath: filepath of database
    Return:
        X: input variables
        Y: target variables
    """
    engine = create_engine("sqlite:///"+database_filepath)
    df = pd.read_sql_table('disaster_dataset',engine)
    X = df.message.values
    Y = df.drop(columns=['id','message','genre','original'])
    return X,Y, Y.columns

def tokenize(text):
    """lower,tokenize,remove stopwords and punctuation and lemmatize tokens
    Args:
        text: the text to clean
    Return:
        list of cleaned tokens"""
    text = text.lower()
    text = word_tokenize(text)
    text = [t for t in text if t not in stop and t not in punctuation]
    text = [lemmatizer.lemmatize(t) for t in text]
    return text


def build_model():
    """Build the model
    Return:
        cv: model with the best params"""
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators':[100, 150],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """display de classification report of each column
    Args:
        model: the fitted model
        X_test: test inputs
        Y_test: test targets
        category_names: categories to test"""
    y_pred = model.predict(X_test)
    Y_test = np.array(Y_test)
    for i in range(y_pred.shape[1]):
        print(classification_report(Y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """save the model
    Args:
        model: model to save
        model_filepath: path to save the model"""
    dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()