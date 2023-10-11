import sys
import nltk
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
import pickle

nltk.download('punkt')

def load_data(database_filepath):
    """
    Load data from a SQLite database.

    Args:
        database_filepath (str): Filepath to the SQLite database.

    Returns:
        X (Series): A Series containing the messages.
        Y (DataFrame): A DataFrame containing the target categories.
        category_names (list): List of category names.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    if not engine.dialect.has_table(engine, "DisasterResponse"):
        raise ValueError("DisasterResponse table does not exist in the database!")
    data = pd.read_sql('SELECT * FROM DisasterResponse', engine)
    X = data.message
    Y = data.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize and lemmatize a text.

    Args:
        text (str): Input text.

    Returns:
        list: List of tokenized and lemmatized words.
    """
    tokens = word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    return [lemmatizer.lemmatize(token).lower().strip() for token in tokens]

def build_model():
    """
    Build a machine learning pipeline.

    Returns:
        pipeline: Machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a machine learning model and print classification reports.

    Args:
        model: Trained machine learning model.
        X_test (Series): Test input data.
        Y_test (DataFrame): Test target data.
        category_names (list): List of category names.
    """
    Y_pred = model.predict(X_test)
    
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        true_labels = Y_test[category]
        predicted_labels = Y_pred[:, i]
        report = classification_report(true_labels, predicted_labels)
        print(report)

def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a pickle file.

    Args:
        model: Trained machine learning model.
        model_filepath (str): Filepath to save the model.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    """
    Main function to train and save a machine learning model.

    Expects command line arguments for the database filepath and model filepath.
    """
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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
