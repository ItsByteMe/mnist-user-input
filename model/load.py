import numpy as np
import keras.models
import tensorflow as tf

from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

def init():
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()

	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")

	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	graph = tf.get_default_graph()
	return loaded_model,graph
