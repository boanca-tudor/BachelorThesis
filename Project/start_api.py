from flask import Flask
from flask_restful import Api
from api.controller.hello import *

app = Flask(__name__)
api = Api(app)

api.add_resource(UploadEndpoint, "/upload")

if __name__ == '__main__':
    app.run(debug=True)