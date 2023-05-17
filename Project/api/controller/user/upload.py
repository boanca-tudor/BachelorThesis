from flask_jwt_extended import jwt_required
from flask_restful import Resource
from flask import request


class UploadEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__holder = kwargs['image_holder']

    @jwt_required()
    def post(self):
        data = request.files['image']
        self.__holder.raw_image = data
        self.__holder.image = data.read()
        return {'message': 'success'}
