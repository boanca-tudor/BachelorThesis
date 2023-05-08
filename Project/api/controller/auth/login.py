import bcrypt
from flask_jwt_extended import create_access_token, create_refresh_token
from flask_restful import Resource
from flask import request, jsonify

from api.model.database import ApplicationUser
from api.response_utils import create_bad_request


class LoginEndpoint(Resource):
    def __init__(self, **kwargs):
        self.__database = kwargs['database']

    def post(self):
        data = request.get_json()
        if not data.get('email') or not data.get('password'):
            return create_bad_request('email/password not found')

        user = self.__database.session.execute(self.__database.select(ApplicationUser)
                                               .where(ApplicationUser.email == data['email'])).scalar()

        if user is None:
            return jsonify(error='Invalid user/password combination')
        given_password = data['password']
        given_password_bytes = given_password.encode('utf-8')
        user_password = user.password
        user_password_bytes = user_password.encode('utf-8')
        if not bcrypt.checkpw(given_password_bytes, user_password_bytes):
            return jsonify(error='Invalid user/password combination')
        access_token = create_access_token(identity=user.username)
        refresh_token = create_refresh_token(identity=user.username)
        return jsonify(access_token=access_token, refresh_token=refresh_token)
