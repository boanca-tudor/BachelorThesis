from io import BytesIO

from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import make_response
from models.face_detector import *


class CropFaceEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']

    @jwt_required()
    def get(self):
        result, image_data = get_cropped_face("image.jpg")
        if result == 0:
            image_data = BytesIO(self.__holder.image)
            response = make_response(image_data.getvalue())
            response.headers['Content-Type'] = 'image/jpg'
            return response
        else:
            image_data = BytesIO(image_data)
            response = make_response(image_data.getvalue())
            response.headers['Content-Type'] = 'image/jpg'
            return response
