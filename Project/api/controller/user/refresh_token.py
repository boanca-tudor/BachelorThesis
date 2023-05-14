from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from flask_restful import Resource
from flask import jsonify


class RefreshTokenEndpoint(Resource):
    @jwt_required(refresh=True)
    def get(self):
        identity = get_jwt_identity()
        access_token = create_access_token(identity=identity)
        return jsonify(access_token=access_token)
