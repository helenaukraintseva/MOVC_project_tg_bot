from fastapi import FastAPI
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Update
from io import BytesIO
from PIL import Image
from models_utils import input_img_to_array, pca_reduce, predict_image, join_files
import pickle
from sklearn.decomposition import PCA
from sklearn.svm import SVC

app = FastAPI()

API_TOKEN = '6921521470:AAHIi_d2An9GRsaHx83ogPw0nVUPQk7uRLs'
WEBHOOK_URL = 'https://tom-and-jerry-bot.onrender.com'

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

# Загрузка модели для классификации
model_path = 'model_svc.pkl'
with open(model_path, 'rb') as m:
    model = pickle.load(m)

# Объединение файлов модели PCA и загрузка
join_files('model_pca.pkl', 'model', 'model_reconstructed.pkl')
pca_transform_path = 'model_reconstructed.pkl'
with open(pca_transform_path, 'rb') as pca:
    pca_transform = pickle.load(pca)

logging.basicConfig(level=logging.INFO)

async def classify_image(image):
    one_d_image = input_img_to_array(image)
    img_pca = pca_reduce(pca_transform, one_d_image)
    predict_class = predict_image(model, img_pca)
    logging.info(f'{predict_class}')
    return predict_class

@app.post('/webhook')
async def handle_update(update: Update):
    await dp.process_update(update)
    return {"ok": True}

@app.on_event("startup")
async def on_startup():
    await bot.set_webhook(WEBHOOK_URL)

@app.on_event("shutdown")
async def on_shutdown():
    await bot.delete_webhook()

@dp.message_handler(commands=['send_photo'])
async def request_photo(message: types.Message):
    await message.reply("Пожалуйста, отправьте фото для классификации.")

@dp.message_handler(content_types=['photo'])
async def handle_photo(message: types.Message):
    photo_bytes = BytesIO()
    await message.photo[-1].download(destination_file=photo_bytes)
    photo_bytes.seek(0)
    image = Image.open(photo_bytes)

    # Вызов функции классификации
    classification_result = await classify_image(image)
    await message.reply(f"Результат классификации: {classification_result}")

