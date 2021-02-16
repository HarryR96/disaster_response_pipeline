import time
start_time = time.time()

import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


def load_data(database_filepath):
    
    """Loads data from a SQLite database file and separates it into features
    and target variables.
    
    Args:
        database_filepath (string): file to read data from
    
    Returns:
        X (list of strings): array of feature variables
        Y (binary array): array of target variables
        category_names (list of strings): names of category labels
    
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('messages', engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    return X, Y, df.columns[4:]


def tokenize(text):
        
    """Normalizes text and separates individual terms into an array of lemmatized, lower case
    tokens without whitespace and with stopwords removed.
    
    Args:
        text (string): text to tokenize
    
    Returns:
       list of strings: tokenized text
    
    """
    
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens if token \
                    not in stopwords.words('english')]
    return clean_tokens


def build_model():
        
    """Pipeline to tokenize and vectorize text data and build multilabel
    classification model.
    
    Args:
        None
    
    Returns:
       classifier: classification model object
    
    """
    
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
        
    """Prints classification evaluation metrics per category label.
    
    Args:
        model (classifier): classification model to evaluate
        X_test (list of strings): features to generate predictions from
        Y_test (binary array): true values of target variables
        category_names (list of strings): names of category labels
    
    Returns:
       None
    
    """
    
    Y_pred = model.predict(X_test)
    Y_test_df = pd.DataFrame(data=Y_test, columns=category_names)
    Y_pred_df = pd.DataFrame(data=Y_pred, columns=category_names)
    for category in category_names:
        print('-'*50)
        print(category)
        print(classification_report(Y_test_df[category], Y_pred_df[category]))
        print('-'*50)


def save_model(model, model_filepath):
        
    """Exports trained model as pickle file.
    
    Args:
        model (classifier): trained classification model
        model_filepath (string): filepath to save model to
    
    Returns:
       None
    
    """
    
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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
        print("--- %s seconds ---" % (time.time() - start_time))

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()