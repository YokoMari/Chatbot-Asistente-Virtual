# Librerias
import torch    # Deep Learning
import random   # Respuestas Aleatorias   
import json     # Archivos JSON

from model import NeuralNet                     # Importa Modelo RNA Personalizada
from nltk_utils import bag_of_words, tokenize   # NLP / Palabras a Vector / Mensaje a Palabras

# Configuracion del Dispositivo (Ejecucion)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Archivo del Entrenamiento / Intenciones del Chatbot
with open('intents.json', 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)              # JSON en un diccionario (Intenciones y Respuestas)

# Datos despues del entrenamiento
FILE = "data.pth"
data = torch.load(FILE)

# Extraer Informacion
# Dimensiones de la Red Cargada
input_size = data["input_size"]                 # #str de Entrada
hidden_size = data["hidden_size"]               # Tamaño Capa Oculta (RNA)
output_size = data["output_size"]               # #Intenciones (Clases)

all_words = data['all_words']                   # Lista Palabras unicas
tags = data['tags']                             # Lista Posibles Etiquetas Intenciones
model_state = data["model_state"]               # Estado del Modelo (pesos)

#Inicializacion del Modelo RNA (ya entrenado)
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)              # Carga los Pesos (entrenamiento)
model.eval()                                    # Solo predicciones (sin entrenamiento)

# Nombre del Chatbot
bot_name = "Tyto"

# Funcion que procesa el Mensaje del Usuario
def get_response(msg):
    sentence = tokenize(msg)                    # Mensaje a lista
    X = bag_of_words(sentence, all_words)       # Palabras a vectores
    X = X.reshape(1, X.shape[0])                # Ajusta el vector
    X = torch.from_numpy(X).to(device)          # Vector a Tensor (@)

    # Proceso del modelo
    output = model(X)                           # Entrada -> Modelo -> Salida
    _, predicted = torch.max(output, dim=1)     # Encuentra Intencion (clase) con mas valor en salida

    # Etiqueta de la Intencion Predicha
    tag = tags[predicted.item()]                # Indice de la Intencion (clase) predicha

    # Calcular Probabilidades
    probs = torch.softmax(output, dim=1)        # Salida a % (0 y 1)
    prob = probs[0][predicted.item()]           # % de la Intencion Predicha

    # Comparar % con un Umbral
    if prob.item() > 0.75:                      # Si la Intencion Predicha es mayor de 75% es confiable
        
        # Responder a la Intencion
        for intent in intents['intents']:                   # Lista Intenciones
            if tag == intent["tag"]:                        # Compra la intencion Predicha
                return random.choice(intent['responses'])   # Respuesta "Aleatoria" de la Intencion Predicha
    
    # Prediccion menor a 75% - Respuesta a Prediccion incierta
    return "Parece que necesito un poco de ayuda para entender tu mensaje." 

# Consola
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

