from io import BytesIO

import numpy as np
import tensorflow as tf
from flask import make_response
from flask_jwt_extended import jwt_required
from flask_restful import Resource
from skimage.io import imsave

from models.model_utils import load_image, create_all_ages


class GenerateEvolutionEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']
        self.__model = kwargs['model']

    @jwt_required()
    def get(self):
        ages = create_all_ages()
        image_data = load_image(self.__holder.image)
        results = []
        image_tensor = tf.expand_dims(image_data, axis=0)
        for age in ages:
            results.append(tf.squeeze(self.__model.eval([image_tensor, age])).numpy())

        results = np.concatenate(results, axis=1)
        imsave("result.jpg", (((results + 1) / 2) * 255).astype(np.uint8))
        image_data = BytesIO(open("result.jpg", "rb").read())
        response = make_response(image_data.getvalue())
        response.headers['Content-Type'] = 'image/jpg'
        return response
