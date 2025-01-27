# Librerias
import numpy as np                          # Calculos matematicos y @
import nltk                                 # NLP
from nltk.stem.porter import PorterStemmer  # Algoritmo para la reducir palabras a su raíz (stemmer)

# Inicializa el stemmer
stemmer = PorterStemmer()

# Funcion descompone una oracion en una lista de palabras o tokens
def tokenize(sentence):
    """
    Un token puede ser una palabra, un carácter de puntuación o un número.
    Ej: sentence = "Fechas de matriculas." ---> ['Fechas', 'de', 'matriculas', '.']
    """
    return nltk.word_tokenize(sentence)     # Divide la oración en palabras individuales

# Funcion Buscar Raiz Base 
def stem(word):
    """
    Stemming = encontrar la forma raíz de la palabra.
    Ej: words = ["organizar", "organizo", "organizara"] ---> words = [stem(w) for w in words] ---> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())       # Reduce la palabra a su raíz en minúsculas

# Funcion Representar numericamente una oracion (Bolsa de Palabras)
def bag_of_words(tokenized_sentence, words):
    """
    Devuelve un array de tipo "bag of words" (bolsa de palabras):
    1 para cada palabra conocida que esté en la oración, 0 en caso contrario.
    Ej: sentence = ["hola", "como", "estas"] ---> words = ["hola", "adios", "gracias", "estas", "bien"] ---> bag = [ 1, 0, 0, 0, 0, 1, 0]
    """
    # Stemmea cada palabra tokenizada (la reduce a su raíz)
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Inicializa un array de ceros del tamaño del número de palabras conocidas
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):         # Para cada palabra en 'words', verifica si está en la oración tokenizada
        if w in sentence_words:             # Si la palabra está en la oración tokenizada
            bag[idx] = 1                    # Asigna 1 en la posición correspondiente en la bolsa de palabras, sino es 0
    return bag                              # Devuelve el array de "bag of words"
