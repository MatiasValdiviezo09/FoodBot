import discord
from discord.ext import commands
import tensorflow as tf
import keras
import numpy as np
from PIL import Image
from io import BytesIO
import asyncio

# Ruta del modelo y etiquetas
MODEL_PATH = r"C:\Users\mvald\Documents\mi_proyecto\modelo\model.savedmodel"
LABELS_PATH = r"C:\Users\mvald\Documents\mi_proyecto\labels.txt"

#  Verificación de la versión de TensorFlow
print(f"Usando TensorFlow versión: {tf.__version__}")

#  Cargar el modelo en formato SavedModel
try:
    model = keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
    print("✅ Modelo cargado con éxito.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {str(e)}")
    exit(1)

#  Cargar etiquetas desde labels.txt
try:
    with open(LABELS_PATH, "r") as f:
        labels = f.read().splitlines()
    print("✅ Etiquetas cargadas correctamente.")
except FileNotFoundError:
    print("❌ Error: labels.txt no encontrado.")
    exit(1)

# 🔹 Inicializa el bot de Discord
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Bot conectado como {bot.user}")

@bot.command()
async def comida(ctx):
    await ctx.send("Envía una imagen de comida para identificarla. 🍔🍕")

    def check(m):
        return m.author == ctx.author and m.attachments

    try:
        # 🔹 Espera un mensaje con imagen adjunta
        msg = await bot.wait_for("message", check=check, timeout=30)
        img_bytes = await msg.attachments[0].read()

        # 🔹 Preprocesamiento de la imagen
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 🔹 Predicción con el modelo (corrección para obtener el valor real)
        prediction_dict = model(img_array)  # TFSMLayer devuelve un diccionario
        prediction = np.array(list(prediction_dict.values())[0])  # Extraer valores numéricos
        
        predicted_label = labels[int(np.argmax(prediction))]  # Asegurar que el índice es entero
        confidence = float(np.max(prediction)) * 100
        confidence = round(confidence, 2)  # Mantiene dos decimales
        if confidence > 99.5:
            confidence = 100.0


        await ctx.send(f"La comida parece ser: **{predicted_label}** 🍽️ con una confianza de {confidence:.2f}%")
    except asyncio.TimeoutError:
        await ctx.send(" No se recibió ninguna imagen a tiempo. Intenta de nuevo.")
    except Exception as e:
        await ctx.send(f" Hubo un error procesando la imagen: {str(e)}")

# 🔹 Ejecutar el bot
TOKEN = "TOKEN_DEL_BOT"  # Reemplaza con tu token de bot
bot.run(TOKEN)