import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import seaborn as sns
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import re



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #print(df.genre)
    
  # Extract categories columns
    columns_categories = df.columns.drop(['id','message','original','genre']).tolist()
    categories_values = df.drop(['id','message','original','genre'], axis=1)
    #print(categories_values.head(1))

    category_counts=[]
    category_names=[]
    #category_counts = pd.DataFrame()
    
    category_counts = categories_values[categories_values == 1].sum(axis=0)
    category_names = columns_categories

    print("category_counts=\n",category_counts)
    print("category_names=\n",category_names)
        
     
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                {
                    "x":category_names,
                    "y":category_counts,
                    "type":"scatter"
                }
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }#,
                
                #'barmode': 'stack'
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()