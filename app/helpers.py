import os
import json 
import io
import requests 
import numpy as np
import tensorflow as tf
from PIL import Image

ASSETS_DIR = "../assets"
TEST_IMAGES_DIR = "../images"

classes = json.load(open(os.path.join(ASSETS_DIR,'class_indices.json')))
classes = {v:k for k,v in classes.items()}

URL = "http://localhost:8501/v1/models/flower-classification:predict"

def read_img_file(file) -> Image.Image:
    img = Image.open(io.BytesIO(file))
    if img.mode == "RGBA":
        img = img.convert("RGB")
    img = img.resize((224,224), Image.ANTIALIAS)
    img = tf.expand_dims(np.asarray(img)/255, 0)
    return img

def load_img(img_path, show:bool=False):
  # img_path = random.choice(glob.glob(f"{DATASET_DIR}/test/{cls}/*.jpg"))
  img = PIL.Image.open(img_path)
  img = img.resize((224,224), PIL.Image.ANTIALIAS)
  if show:
    plt.imshow(img)
    plt.title(cls)
  img = tf.expand_dims(np.asarray(img)/255, 0)
  return img

def predict(img):
  headers = {
      "content-type" : "application/json"
  }
  data = json.dumps({
      "signature_name": "serving_default", "instances": img.numpy().tolist()
  })
  response = requests.post(URL, data=data, headers=headers)
  response_data = json.loads(response.text)['predictions'][0]
  confidence = np.max(response_data)
  predicted_class = classes[np.argmax(response_data)]
  return response_data,confidence,predicted_class