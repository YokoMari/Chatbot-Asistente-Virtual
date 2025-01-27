# Librerias
from flask import Flask, render_template, request, jsonify # Crear Apps web
from chat import get_response                              # Procesa Chats (mensajes)

# Inicializacion de la app Flask
app = Flask(__name__)

# Ruta que responde a Get
@app.get("/")
def index_get():
    return render_template ("index.html")       # Renderiza el HTML

# Ruta que responde a Post (Usuario envia el mensaje y se genera una respuesta)
@app.post("/predict")
def predict():                                  # Metodo de respuesta del Chatbot
    text = request.get_json().get("message")    # Obtiene la pregunta
    response = get_response(text)               # Funcion con el mensaje del Usuario para obtener una respuesta
    message = {"answer": response}              # Diccionario con la respuesta
    return jsonify(message)                     # Diccionario a JSON y envia la respuesta al usuario

# Ejecutar App
if __name__ == "__main__":
    app.run(debug=True)