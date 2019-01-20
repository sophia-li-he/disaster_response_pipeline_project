import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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
    df_top5 = df.iloc[:, 4:].sum().sort_values(ascending=False).head().to_frame()
    df_top5.columns = ['count']
    df_bottom5 = df.iloc[:, 4:].sum().sort_values(ascending=True).head().to_frame()
    df_bottom5.columns = ['count']
    top5_list = df_top5.index.tolist()
    top5_count = df_top5['count']
    bottom5_list = df_bottom5.index.tolist()
    bottom5_count = df_bottom5['count']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
	                    'title': "Genre"
	                }
	            }
	        },
	            {
	            'data': [
	                Bar(
	                    x=top5_list,
	                    y=top5_count
	                )
	            ],

	            'layout': {
	                'title': 'Counts of Most Frequent Message Categories',
	                'yaxis': {
	                    'title': "Count"
	                },
	                'xaxis': {
	                    'title': "Message Category"
	                }
	            }
	        },
	       {
	            'data': [
	                Bar(
	                    x=bottom5_list,
	                    y=bottom5_count
	                )
	            ],

	            'layout': {
	                'title': 'Counts of Least Frequent Message Categories',
	                'yaxis': {
	                    'title': "Count"
	                },
	                'xaxis': {
	                    'title': "Message Category"
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