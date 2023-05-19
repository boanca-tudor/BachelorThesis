from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import make_response
from io import BytesIO


class GetUploadedEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']

    @jwt_required()
    def get(self):
        image_data = self.__holder.image
        response = make_response(image_data.getvalue())
        response.headers['Content-Type'] = 'image/jpg'
        return response
