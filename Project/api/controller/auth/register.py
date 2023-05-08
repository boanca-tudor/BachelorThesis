from flask_restful import Resource
from flask import request, jsonify
import bcrypt

from api.model.database import ApplicationUser


class RegisterEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__db = kwargs['database']

    def post(self):
        body = request.get_json()
        password = body['password']
        password_bytes = password.encode('utf-8')
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password_bytes, salt).decode('utf-8')
        new_user = ApplicationUser(username=body['username'], email=body['email'], password=password_hash)
        self.__db.session.add(new_user)
        self.__db.session.commit()
        return jsonify(username=new_user.username)
