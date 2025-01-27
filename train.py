# Librerias
import random                                       # Respuestas Aleatorias 
import json                                         # Archivos JSON
import torch                                        # Deep Learning
import torch.nn as nn                               # Modulo RNA
import numpy as np                                  # Calculos matematicos y @

from torch.utils.data import Dataset, DataLoader    # Manejo de conjunto de datos y cargarlos
from nltk_utils import bag_of_words, tokenize, stem # Funciones de procesamiento de Texto (NLP/Palabras a Vector/Mensaje a Palabras/descomponer palabras)
from model import NeuralNet                         # Importa Modelo RNA Personalizada

# Leyendo el Json (Intenciones)
with open('intents.json', 'r') as f:
    intents = json.load(f)              # JSON a un diccionario

# Inicializacion de Listas
all_words = []                          # Lista Palabras Unicas
tags = []                               # Lista etiquetas de Intencion
xy = []                                 # Lista pares de Patrones (frases)

# Iterar a travez de cada patron de Intencion (JSON)
for intent in intents['intents']:
    tag = intent['tag']                 # Obtiene la etiqueta de intención
    tags.append(tag)                    # Agrega la etiqueta a la lista de etiquetas

    for pattern in intent['patterns']:  # Itera a través de los patrones de conversación
        w = tokenize(pattern)           # Tokeniza cada patrón en palabras
        all_words.extend(w)             # Agrega las palabras a la lista de todas las palabras unicas
        xy.append((w, tag))             # Agrega el par (palabras, etiqueta) a la lista de patrones

# Aplica stemming y convierte las palabras a minúsculas
ignore_words = ['?', '.', '!']          # Palabras que se ignorarán
all_words = [stem(w) for w in all_words if w not in ignore_words]   # Reduce las palabras a su raíz

# Convierte la lista en un conjunto para eliminar duplicados y ordena
all_words = sorted(set(all_words))      # para Palabras Unicas
tags = sorted(set(tags))                # para Etiquetas

# Imprimir Resultados
print(len(xy), "patterns")              # Cuántos patrones encontrados
print(len(tags), "tags:", tags)         # Cuántas etiquetas únicas hay
print(len(all_words), "unique stemmed words:", all_words)   # Cuántas palabras únicas hay

# Creación de Datos de Entrenamiento
X_train = []                            # Lista para las características (X)
y_train = []                            # Lista para las etiquetas (y)

#Iterar sobre los pares de xy
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words) # Oracion a bolsa de palabras
    X_train.append(bag)                 # Agrega la bolsa de palabras a la lista de características
    label = tags.index(tag)             # Obtiene el índice de la etiqueta en la lista de etiquetas
    y_train.append(label)               # Agrega el índice de la etiqueta a la lista de etiquetas

# Convierte las listas de Python a arreglos de NumPy
X_train = np.array(X_train)             # Lista Caracteristicas
y_train = np.array(y_train)             # Lista Etiquetas

# Hiperparámetros (Valores de Config.)
num_epochs = 1000                       # #veces que se recorre los datos
batch_size = 8                          # Tamaño de muestras para el entrenamiento
learning_rate = 0.001                   # Tasa de aprendizaje (Velocidad Ajuste de Pesos)
input_size = len(X_train[0])            # #Caracteristicas de entreda (BOW)
hidden_size = 8                         # Tamaño la Capa Oculta
output_size = len(tags)                 # Tamaño de la Salida (#Etiquetas Unicas-Intenciones)
print(input_size, output_size)

# Clase Preparar los datos del Entrenamiento
class ChatDataset(Dataset):

    # Inicializa las variables
    def __init__(self):
        self.n_samples = len(X_train)   # #Total de muestras (entrada) (oracion y etiqueta)
        self.x_data = X_train           # Datos de entrada (características) (BOW)
        self.y_data = y_train           # Etiquetas (clases o intenciones) (de la entrada)

    # Acceder a una muestra especifica
    # Soporta indexación para que dataset[i] se pueda usar para obtener la i-ésima muestra
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]   # Devuelve el par (entrada, salida)

    # Puede llamar a len(dataset) para obtener el tamaño
    def __len__(self):
        return self.n_samples           # Devuelve el número de muestras

#Dividir en lotes
dataset = ChatDataset()                 # Crea una instancia del conjunto de datos
# Crea un DataLoader para gestionar el acceso a los datos
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,    # #Muestras se Procesan a la vez
                          shuffle=True,             # Mezcla aleatoriamente 
                          num_workers=0)            # Número de hilos (procesamiento)

# Establece el dispositivo 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Crea una instancia del modelo RNA personalizado - (Tamaño entrada (BOW)-(Tamaño Capa Oculta)-(#ClasesIntenciones)
model = NeuralNet(input_size, hidden_size, output_size).to(device)  # Mueve el modelo al dispositivo

# Define la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Entrenamiento del Modelo
for epoch in range(num_epochs):                     # Itera a través de cada recorrido
    for (words, labels) in train_loader:            # Itera a través de los Lotes de Datos (DataLoader)
        words = words.to(device)                    # Mueve las caracteristicas al dispositivo
        labels = labels.to(dtype=torch.long).to(device) # Mueve las etiquetas al dispositivo
        
        # Paso hacia adelante: Proceso donde se alimentan los datos de entrada y calcula predicciones
        outputs = model(words)                      # Calcula la salida del modelo / Pasa los datos por el Modelo
        loss = criterion(outputs, labels)           # Calcula la pérdida comparando predicciones (salidas) con etiquetas
        
        # Paso hacia atrás y Optimizacion: Proceso de retropropagacion donde se calcula gradientes 
        optimizer.zero_grad()                       # Resetea los gradientes
        loss.backward()                             # Calcula los gradientes / Propagacion del error hacia atras
        optimizer.step()                            # Actualiza los pesos del modelo basándose en los gradientes

    # Imprime la pérdida cada 100 epocas (recorridos)
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Muestra la pérdida actual


# Imprime la pérdida final después de todas las épocas (recorridos)
print(f'final loss: {loss.item():.4f}')

# Guarda el estado del modelo y la información de entrenamiento en un diccionario
data = {
"model_state": model.state_dict(),                  # Guarda los parámetros entrenados del modelo (pesos y sesgos)
"input_size": input_size,                           # Guarda el tamaño de la entrada
"hidden_size": hidden_size,                         # Guarda el tamaño de la capa oculta
"output_size": output_size,                         # Guarda el tamaño de la salida (número de clases o etiquetas)
"all_words": all_words,                             # Guarda la lista de todas las palabras unicas conocidas por el modelo (vocabulario)
"tags": tags                                        # Guarda la lista de las etiquetas unicas (intents) que el modelo ha aprendido a predecir
}

# Archivo donde se guardan los datos del modelo
FILE = "data.pth"
torch.save(data, FILE)

# Mensaje de entrenamiento terminado y guardado
print(f'Entrenamiento completo (~‾▿‾)~. Archivo guardado en {FILE}')
