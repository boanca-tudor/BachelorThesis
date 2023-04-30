from flask_restful import Resource
from flask import request, jsonify


class HelloWorld(Resource):
    def get(self):
        return {'message': 'hello world'}

    def post(self):
        data = request.get_json()
        return jsonify({'object': data})
