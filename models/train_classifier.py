import sys
# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# import statements
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from tqdm import tqdm
import pickle

class TextLenghExtractor(BaseEstimator, TransformerMixin):
    """
    Class for defining an extractor for calculating the lenghts of a strings array.
    
    """

    def fit(self, X, y=None):
        """
        Function for fitting the transformer

        Parameters: self is the same extractor
                    X is an array of strings
        Return:     The extractor
        """
        return self

    def transform(self, text):
        """
        Function for calculating the lenghts of an array of strings

        Parameters: self is the same extractor
                    text is an 2D array
        Return:     lenghts The extracted lenghts of each array in the 2D array
        """
        #print(text.shape)
        lens = []
        for x in text:
            lens.append(len(x))
            #print(len(x), lens)
        lengths = np.asarray(lens).reshape(-1,1)
        return lengths


def display_results(y_test, y_pred):
    """
    Function to display the predicted values with the pipeline, and reports the scores for the
    multiclass-multioutput-multilabel classifier: f1 score, precision and recall
    
    Parameters: y_test is the vector with the testing labels
                y_pred is the vector with the predicted labels
    """
    labels = np.unique(y_pred)
    y_test2 = np.array(y_test)
    y_pred2 = np.array(y_pred)
    
    columnsTest = y_test.shape[1]
    columnsPred = y_pred.shape[1]
    
    # Obtain the precision, recall, and F1 metrics for each feature in y_pred
        
    if columnsTest == columnsPred:
        for i in range(columnsTest):
            print("Category of message= ",y_test.columns[i])
            print(metrics.classification_report(y_test2[:,i],y_pred2[:,i]))
    else:
        print("Columns number in y_test and y_pred are different.")
    return

def load_data(database_filepath):
    """
    Function to load the data from a database with the messages already cleaned
    
    Parameters: database_filepath is the filepaht name of the database
    """
    # load data from database
    database_name = "sqlite:///"+ database_filepath # Get the database name from the IO
    print(database_name)

    engine = create_engine(database_name)
    df = pd.read_sql_table('DisasterResponse', database_name)
    #df.head()

    # Extract messages
    X = df[['id','message','original','genre']]

    columns_categories = df.columns.drop(['id','message','original','genre'])
    print(columns_categories)

    # Extract categories columns
    Y = df[columns_categories]
    return X,Y,columns_categories


def tokenize(text):
    """
    This function obtain the tokens from a string.
    
    Parameters: text is the string from the tokens are obtained
    Return:     clean_tokens as the list of tokens 
    """
    
    # Normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Tokenize text
    tokens = word_tokenize(text)
    #print("\nTokens=", tokens)
    
    # Remove stop words and lemmatize
    #tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    #print("\nStop words removal and lematizer =", clean_tokens)
    
    return clean_tokens

def build_model(X_train):
    """
    This function build a model with a gridsearch to test hyper-parameters for the ExtraTrees and K-Nearest          
     classifiers
    
    Parameters: X_train is the train data to obtain the messages
    Return:     cv is the created model
    """
    textlen = TextLenghExtractor()

    pipeline = Pipeline([
            ('features', FeatureUnion([ 
               ('nlp_pipeline', Pipeline([ ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer()) ]) ), 
               ('textlen', textlen ),
                ]) #End of duplas list for FeatureUnion
            ),
            ('estimator', MultiOutputClassifier(ExtraTreesClassifier(random_state=0, bootstrap=True, max_depth=3))),
            #('estimator', MultiOutputClassifier(RandomForestClassifier())),
            
            ])

    #pipeline.fit(X_train, y_train)

    parameters = [
                    {
                    'features__nlp_pipeline__vect__ngram_range':[(1,2)],          # Allow unigrams, bigrams or both.
                    'features__nlp_pipeline__tfidf__norm':['l2'],                 # Test if l1, l2 or None train better
                    'estimator':[MultiOutputClassifier(ExtraTreesClassifier(random_state=0, bootstrap=True, max_depth=3))],
                    'estimator__estimator__n_estimators': [10],
                    },
                    {
                    'features__nlp_pipeline__vect__ngram_range':[(1,2)],          # Allow unigrams, bigrams or both.
                    'features__nlp_pipeline__tfidf__norm':['l2'],                 # Test if l1, l2 or None train better
                'estimator':[MultiOutputClassifier(RandomForestClassifier())],
                'estimator__estimator__n_estimators': [8],
                   }
                ]


    cv = GridSearchCV(estimator=pipeline, param_grid=parameters,refit=True,verbose=2,n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates a model by predicting with the test data and then displaying the precision, recall, 
    and the f1-score for each category of the messages. The best hyperparameters found for the best model are also          displayed.
    
    Parameters: model is the trained model
                X_test is the train data
                Y_test is the test data 
                category_names is the categories names for the messages, and a message can have more than one

    Return:     None 
    """
    # Print out the best hyperparameters results
    print("Best parameter (CV score=%0.3f):" % model.best_score_)
    print(model.best_params_)

    # Make the prediction for the testing data
    y_pred = model.predict(X_test)

    # Display results, reporting the f1 score, precision and recall for each 
    # output category of the dataset
    display_results(Y_test, y_pred)    
    pass


def save_model(model, model_filepath):
    """
    This function saves the model into a pickle file.
    
    Parameters: model is the trained model
                model_filepath is the name of the pickle file where the model is saved into.
    Return:     None 
    """
    # Exporting the model to a file

    pickle.dump(model, open(model_filepath,'wb'))    
    pass


def main():
    """
    This program upload data from a database, train a model (ExtraTrees and K-nearest classifiers) with the best hyper-parameters, test the model and then save it into a pickle file.
    
    Parameters: None
    Return:     None 
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X.message, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train)
        
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