from flask_restful import Resource
from flask import request, jsonify


class UploadEndpoint(Resource):
    def post(self):
        data = request.files['image']
        data.save('image.jpg')
        return {'message': 'bravo boss'}
