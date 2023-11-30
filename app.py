import pickle
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from textblob import Word
from flask import Flask, render_template, request
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download('punkt')

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None

    if request.method == 'POST':
        post_text = request.form.get('post_text', "")
        friends = int(request.form.get('friends', 0))
        favourites = int(request.form.get('favourites', 0))
        statuses = int(request.form.get('statuses', 0))
        retweets = int(request.form.get('retweets', 0))
        week_day = int(request.form.get('week', 0))
        year = int(request.form.get('year', 0))
        month = int(request.form.get('month', 0))
        day = int(request.form.get('day', 0))

        print("-"*45)
        print(post_text)

        post_text = " ".join(x.lower() for x in post_text.split())

        post_text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", post_text)
        post_text = re.sub(r'''[^a-zA-Z]''', " ", post_text)

        sw = stopwords.words("english")
        post_text = " ".join(x for x in post_text.split() if x not in sw)

        post_text = " ".join([Word(post_text).lemmatize()])

        print("*"*45)
        print(post_text)

        new_data = pd.DataFrame({
            'post_text': [post_text],
            'friends': [friends],  # Proporciona valores para las otras columnas según sea necesario
            'favourites': [favourites],
            'statuses': [statuses],
            'retweets': [retweets],
            'weekday': [week_day],
            'month': [month],
            'day': [day],
            'year': [year]
        })

        print("*"*30)

        print(new_data)

        # Abre el modelo y realiza la predicción
        with open('mental_health.pkl', 'rb') as archivo:
            model = pickle.load(archivo)
            with open('pipeline.pkl', 'rb') as file:
                loaded_pipeline, X_test_transformed_loaded, y_test_loaded = pickle.load(file)
                X_transformed = loaded_pipeline.transform(new_data[['post_text', 'friends', 'favourites', 'statuses', 'retweets', 'weekday', 'month', 'day', 'year']])
                result = model.predict(X_transformed)
                print("result ---------> ", result[0])
                resultado = result[0]

    return render_template('main.html', resultado=resultado)


@app.route('/about')
def about():
    return 'About'


if __name__ == '__main__':
    app.run(debug=True)
