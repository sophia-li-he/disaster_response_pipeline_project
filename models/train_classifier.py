import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download('punkt', 'wordnet', 'stopwords')
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.multioutput import MultiOutputClassifier
import warnings
warnings.simplefilter('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def load_data(database_filepath):
    """
    The function loads the processed data from the database
    Input: database filepath
    Output: the cleaned message variable X, the labels for the message Y, the category names (label)
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('DisasterResponse', engine)
    df = df.iloc[0:3000,]
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()


def tokenize(text):
    """
    The function normalizes the string, removes the punctuation characters, tokenizes and lemmatizes the string
    Input: string
    Output: clean tokens
    """
    lemmatizer = WordNetLemmatizer()
    # Case Normalization
    text = text.lower()
    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    text = word_tokenize(text)
    clean_tokens = []
    for tok in text:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    # remove stop words
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]
    return clean_tokens


def build_model():
    """
    The function builds a machine learning pipeline that vectorizes the input data, uses Grid Search to find
    the best parameters in Random Forest classification model.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [100, 200]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function evalues the model using testing set and output the classification report 
    for each label category
    Input: estimator(model), testing set, testing labels, the category for the lables
    Output: the classification report (precision, recall, f1-score) for each category
    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names, digits=3))


def save_model(model, model_filepath):
    """
    The function saves the model to the file location specified by the user
    Input: model to save, file path
    Output: none
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    file.close()


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