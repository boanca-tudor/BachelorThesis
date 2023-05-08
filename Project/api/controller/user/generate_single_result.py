from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import request
from model_utils import load_image
import tensorflow as tf
import matplotlib.pyplot as plt


class GenerateSingleResultEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']
        self.__model = kwargs['model']

    @jwt_required()
    def get(self):
        age = request.args.get("requiredAge")
        image_data = load_image("image.jpg")
        image = tf.expand_dims(image_data, axis=0)
        result = tf.squeeze(self.__model.eval([image, age])).numpy()
        plt.imshow(result)
        plt.show()
        return {'message': 'ahem'}
