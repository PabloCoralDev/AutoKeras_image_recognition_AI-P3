import json
import autokeras as ak
import tensorflow as tf

#change these ase required
MODEL_FILE = "results/autokeras_image_classifier.keras" #autokeras image classifier [.keras]
LABEL_FILE = "results/class_names.json" #class names json [.json]
IMAGE_FILE = "predict_image/current.jpeg" #load path to current image [will try to improve this in future] !!! THIS COULD BE A .JPG OR .JPEG 
IMG_SIZE = 224

with open(LABEL_FILE) as f:
    classes = json.load(f)["classes"]

model = tf.keras.models.load_model(MODEL_FILE, custom_objects=ak.CUSTOM_OBJECTS) #load existing model in files [must have run maintrainer.py first]

image_bytes = tf.io.read_file(IMAGE_FILE)

#standardize to same preprocessing as training
image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
image = tf.cast(image, tf.float32) / 255.0
image = tf.expand_dims(image, 0)

probs = model.predict(image, verbose=0)[0]
idx = int(probs.argmax())

print(f"{classes[idx]}, with a certainty of {round(float(probs[idx]),3)*100}%")
