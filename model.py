# Librerias 
import torch                # Deep Learning
import torch.nn as nn       # Modulo RNA

# RNA simple (3) Capas
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        # Capas = (Input-#strEntrada) (Hidden-#neuronasCOculta) (num-#clasesSalida)
        self.l1 = nn.Linear(input_size, hidden_size)  # Entrada -> Oculta
        self.l2 = nn.Linear(hidden_size, hidden_size) # Oculta -> Oculta
        self.l3 = nn.Linear(hidden_size, num_classes) # Oculta -> Salida

        # Funcion de Activacion
        self.relu = nn.ReLU()
    
    # Metodo Flujo de Datos de la RNA
    def forward(self, x):

        # Procesamiento de la RNA
        out = self.l1(x)        # Entra 1ra Capa O.
        out = self.relu(out)    # Aplica la Func.Activ.
        out = self.l2(out)      # Entra 2da Capa O.
        out = self.relu(out)    # Aplica la Func.Activ.
        out = self.l3(out)      # Sale Capa Salida

        # No se aplica ninguna función de activación en la salida final
        return out
