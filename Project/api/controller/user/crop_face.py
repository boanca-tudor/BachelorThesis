from io import BytesIO

from PIL import Image
from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import make_response
from models.face_detector import *


class CropFaceEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']

    @jwt_required()
    def get(self):
        image_data = BytesIO(self.__holder.image)
        result, image_data = get_cropped_face(image_data.getvalue())
        if result == 0:
            response = make_response(image_data.getvalue())
            response.headers['Content-Type'] = 'image/jpg'
            return response
        else:
            image_crop_pil = Image.fromarray(image_data)
            bytes_io = BytesIO()
            image_crop_pil.save(bytes_io, format='jpeg')
            response = make_response(bytes_io.getvalue())
            response.headers['Content-Type'] = 'image/jpg'
            return response
