import pickle
import re
import pandas as pd
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
        date = request.form.get('date', "1-2020/1/15")

        print("-"*45)
        print(post_text)

        # Extrae el número correspondiente al día de la semana
        week_day = int(date.split('-')[0])

        # Extrae el resto de la cadena para obtener el año, mes y día
        resto_cadena = date.split('-')[1]

        # Convierte el resto de la cadena en un objeto datetime
        fecha_obj = datetime.strptime(resto_cadena, "%Y/%m/%d")

        # Extrae la información como enteros
        year = int(fecha_obj.year)
        month = int(fecha_obj.month)
        day = int(fecha_obj.day)

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

        transformer = ColumnTransformer(
            transformers=[
                ('post_text', CountVectorizer(), 'post_text')
            ],
            remainder='passthrough'  # Mantiene las columnas no especificadas sin cambios
        )

        # Crea un pipeline con el transformador y, opcionalmente, otros pasos del preprocesamiento
        pipeline = Pipeline([
            ('vectorizer', transformer)
        ])

        # Aplica el pipeline al conjunto completo de datos
        X_transformed = pipeline.fit_transform(new_data[['post_text', 'friends', 'favourites', 'statuses', 'retweets', 'weekday', 'month', 'day', 'year']])

        # Abre el modelo y realiza la predicción
        with open('mental_health.pkl', 'rb') as archivo:
            model = pickle.load(archivo)
            result = model.predict(X_transformed)
            print("result ---------> ", result[0])
            resultado = f"{result[0]} - Mensaje ingresado: {post_text}"

    return render_template('main.html', resultado=resultado)


@app.route('/about')
def about():
    return 'About'


if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)
