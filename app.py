from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
from keras.applications import inception_v3
from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator as IDG
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_ARCH = 'models/your_model.json'
MODEL_WEIGHTS = 'models/your_model.h5'
MODEL = 'models/galana.hdf5'

# # Model reconstruction from JSON file
# with open(MODEL_ARCH, 'r') as f:
#     model = model_from_json(f.read())

# # Load weights into the new model
# model.load_weights(MODEL_WEIGHTS)


# base_model = inception_v3.InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=[200, 200, 3])

# model = Sequential([
#     base_model,
#     Dense(1024, activation='relu'),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dense(4, activation='softmax')
# ])

# for layer in base_model.layers:
#     layer.trainable = False

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(lr=0.000001),
#               metrics=['accuracy'])


model = load_model(MODEL)

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200, 200))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print("Preds: ")
        print(preds)
        # Process your result for human
        labels = ['Spiral', 'Elliptical', 'Irregular', 'Other']
        pred_class = preds.argmax(axis=-1)            # Simple argmax
        return sorted(labels)[pred_class[0]]
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
