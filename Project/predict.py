import pickle
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import numpy as np
import pandas as pd
import requests

from build_models import get_tfidf
from clean import strip, to_lower, lemmatize_words, rm_numbers

app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorizer.pk', 'rb'))

class PredictSentiment(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        # vectorize the user's query and make a prediction
        input = pd.DataFrame([user_query], columns=['Text'])
        input = clean(input)
        feat = vectorizer.transform(input.text_cleaned.astype('U').tolist())

        prediction = loaded_model.predict(feat)
        pred_proba = loaded_model.predict_proba(feat)
        print(pred_proba)
            
        # round the predict proba value and set to new variable
        confidence = max(pred_proba[0]) #round(pred_proba[0], 3)

        # create JSON object
        output = str('prediction: '+ prediction[0] + ', and confidence: ' + str(round(confidence,3)))
        
        return output

def clean(df):

    textdata = df['Text']

    print("Made it to data cleaning")
    textdata = strip(textdata); print("stripped")
    textdata = to_lower(textdata); print("all lowercase")
    textdata = rm_numbers(textdata); print('removed numbers')
    textdata = lemmatize_words(textdata); print('lemmatized words') # chose lemmatizing bc it is less aggressive + makes more sense

    df['text_cleaned'] = textdata
    df['text_cleaned'] = df.text_cleaned.apply(' '.join)
    df = df[['text_cleaned']]

    print("Finished cleaning data.")

    return df

api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=False)

    # url = 'http://127.0.0.1:5000/'
    # params ={'query': 'that movie was boring'}
    # response = requests.get(url, params)
    # response.json()
    # predict_line("this new purse is great. I love that it is so spacious and that it helps me to organize my things.")