import json

from flask import make_response


def create_bad_request(message: str):
    response = make_response(json.dumps({'error': message}), 400)
    response.headers['Content-Type'] = 'application/json'
    return response
