from flask import Flask, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    with open('modelo_titanic.pkl', 'rb') as archivo:
        model = pickle.load(archivo)
        result = model.predict([[20, 0.4, 3.0, 0]])
        print(result[0])
        return render_template('main.html', resultado=result[0])


@app.route('/about')
def about():
    return 'About'


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)
