# Clasificador de Dígitos MNIST con PyTorch

Este repositorio contiene una implementación sencilla en PyTorch para clasificar dígitos manuscritos del conjunto de datos MNIST. El proyecto está estructurado para entrenar una red neuronal densa (Perceptrón Multicapa), guardar los pesos aprendidos en disco y reutilizarlos en ejecuciones posteriores sin necesidad de volver a entrenar.

## Descripción del Proyecto

El objetivo de este proyecto es servir como una base clara para entender los conceptos fundamentales de PyTorch:

* Procesamiento de datos desde archivos CSV con Pandas.
* Conversión de datos a tensores y uso de `DataLoader` para manejar minibaches.
* Definición de una arquitectura de red neuronal mediante `nn.Module`.
* Persistencia del modelo (`torch.save` y `torch.load`) usando `state_dict`.
* Evaluación del modelo en modo inferencia (`model.eval()` y `torch.no_grad()`).

## Arquitectura de la Red (`RedMNIST`)

La red toma como entrada la imagen aplanada de 28x28 píxeles (784 valores) y pasa por las siguientes capas:

1. **Capa de entrada:** `nn.Linear(784, 128)` seguida de activación `ReLU`.
2. **Capa oculta:** `nn.Linear(128, 64)` seguida de activación `ReLU`.
3. **Capa de salida:** `nn.Linear(64, 10)` que produce las puntuaciones (logits) para las clases del 0 al 9.

## Estructura del Archivo de Datos

El script espera encontrar el archivo `mnist_test.csv` en la raíz del proyecto con el siguiente formato:
* Primera columna: Etiqueta real del número (entero del 0 al 9).
* Columnas restates (1 a 784): Valores de los píxeles normalizados entre 0 y 255.