Bot de Discord para Clasificación de Imágenes de Comida

Este proyecto implementa un bot de Discord que utiliza un modelo de Machine Learning entrenado con Teachable Machine para clasificar imágenes de comida. 🥗🍕🍔

Características
- Identifica tres categorías: Pizza, Hamburguesa y Ensalada.
- Responde a las imágenes enviadas por usuarios en Discord.
- Construido con TensorFlow, Keras y Discord.py.

¿Cómo funciona?
El modelo de clasificación fue entrenado utilizando Teachable Machine, exportado en formato TensorFlow (SavedModel) y luego integrado al bot utilizando Keras y Python. El bot procesa las imágenes enviadas por los usuarios y responde con la categoría más probable junto con la confianza porcentual del modelo.

Instalación
1. Clona este repositorio:
   ```sh
   git clone https://github.com/TU-USUARIO/Bot-Clasificador-Comida.git
   cd Bot-Clasificador-Comida