Reconocimiento de Dígitos con Redes Neuronales
Este proyecto implementa un sistema de reconocimiento de dígitos manuscritos utilizando redes neuronales convolucionales. El sistema es capaz de clasificar dígitos del 0 al 9 a partir de imágenes, tanto del dataset MNIST como de imágenes personalizadas capturadas por cámara.

Objetivo del proyecto
Desarrollar y entrenar un modelo de redes neuronales para clasificación de dígitos, evaluar su rendimiento en diferentes conjuntos de datos, e implementar un sistema de predicción en tiempo real que permita reconocer dígitos a través de la cámara web.

Archivos del proyecto
Análisis y entrenamiento del modelo → [A3_2_Redes_Neuronales.ipynb](./A3_2_Redes_Neuronales.ipynb)

Versión HTML del análisis → [A3_2_Redes_Neuronales.html](./A3_2_Redes_Neuronales.html)

Modelo entrenado → [my_model.h5](./my_model.h5)

Script de predicción en tiempo real → [digits.py](./digits.py)

Documentación del proyecto → [README.md](./README.md)

Metodología implementada
Entrenamiento del modelo con el dataset MNIST

Evaluación del rendimiento en conjuntos de prueba y validación

Procesamiento de imágenes personalizadas para adaptarlas al modelo

Implementación de sistema en tiempo real para reconocimiento mediante cámara

Análisis comparativo del rendimiento en diferentes escenarios

Características técnicas
Arquitectura: Red neuronal fully connected con capas densas

Preprocesamiento: Binarización adaptativa y normalización de imágenes

Métricas: Exactitud, matrices de confusión y análisis de errores

Implementación: Sistema en tiempo real con OpenCV y TensorFlow
