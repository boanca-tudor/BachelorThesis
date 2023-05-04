from flask_restful import Resource
from flask import request

from api.repository.image_holder import ImageHolder


class UploadEndpoint(Resource):
    def __init__(self, image_holder: ImageHolder):
        self.__holder = image_holder

    def post(self):
        data = request.files['image']
        self.__holder.image = data
        return {'message': 'bravo boss'}
