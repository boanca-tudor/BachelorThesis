from io import BytesIO

from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import request, make_response
from model_utils import load_image, create_all_ages
import tensorflow as tf
from skimage.io import imsave
import numpy as np


class GenerateEvolutionEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']
        self.__model = kwargs['model']

    @jwt_required()
    def get(self):
        ages = create_all_ages()
        image_data = self.__holder.image.read()
        image = tf.expand_dims(image_data, axis=0)
        results = [image]
        for age in ages:
            image_tensor = tf.expand_dims(image, axis=0)
            results.append(tf.squeeze(self.__model.eval([image_tensor, age])).numpy())

        results = np.concatenate(results, axis=1)
        imsave("result.jpg", (((results + 1) / 2) * 255).astype(np.uint8))
        image_data = BytesIO(open("result.jpg", "rb").read())
        response = make_response(image_data.getvalue())
        response.headers['Content-Type'] = 'image/jpg'
        return response
