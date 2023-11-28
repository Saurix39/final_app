import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None

    if request.method == 'POST':
        mensaje = request.form['mensaje']

        # Abre el modelo y realiza la predicción
        with open('modelo_titanic.pkl', 'rb') as archivo:
            model = pickle.load(archivo)
            result = model.predict([[20, 0.4, 3.0, 0]])
            print("result ---------> ",result[0])
            resultado = f"{result[0]} - Mensaje ingresado: {mensaje}"

    return render_template('main.html', resultado=resultado)

@app.route('/about')
def about():
    return 'About'

if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8080, debug=True)