import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
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
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    df_melt = pd.melt(df, id_vars='id', value_vars=df.columns[5:], var_name='category', \
                      value_name='count_message')
    category_counts = df_melt.groupby('category').count_message.sum().reset_index()
    new_df = df.copy()
    new_df['sum'] = new_df[new_df.columns[4:]].sum(axis=1)
    new_df['labeled'] = new_df['sum'].apply(lambda x: 0 if x==0 else 1)
    label_counts = new_df.groupby('labeled').id.count().reset_index()
    label_counts = label_counts.sort_values(by='labeled', ascending=False).reset_index()
    label_counts.rename(columns={'id': 'count_message'}, inplace=True)
    label_counts['labeled'] = label_counts['labeled'].map({0: 'unlabeled', 1: 'labeled'})
    
    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    values=genre_counts,
                    labels=genre_names,
                    marker={'colors': ['#a2c1ec', '#e4ab7c', '#96d572']}
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres'
            }
        },
        {
            'data': [
                Bar(
                    x=label_counts['labeled'],
                    y=label_counts['count_message'],
                    marker={'color': '#3f7222'}
                )
            ],
            
            'layout': {
                'title': 'Labeled vs Unlabaled Messages',
                'yaxis': {'title': '# of Messages'}
            }
        },
        {
            'data': [
                Bar(
                    x=category_counts['category'],
                    y=category_counts['count_message'],
                    marker={'color': '#223f72'}
                )
            ],
            
            'layout': {
                'title': 'Count of Messages per Category',
                'yaxis': {'title': '# of Messages'}
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