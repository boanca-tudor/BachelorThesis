from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_restful import Api

from api.controller.auth.register import RegisterEndpoint
from api.controller.auth.login import LoginEndpoint
from api.controller.user.crop_face import CropFaceEndpoint
from api.controller.user.generate_evolution import GenerateEvolutionEndpoint
from api.controller.user.generate_single_result import GenerateSingleResultEndpoint
from api.controller.user.get_uploaded import GetUploadedEndpoint
from api.controller.user.refresh_token import RefreshTokenEndpoint
from api.controller.user.upload import *
from api.repository.image_holder import ImageHolder
from api.model.database import init_db
from config_utils import *
from models.caae import CAAE


model = CAAE(z_channels=100,
             l_channels=10,
             gen_channels=1024)
checkpoint_dir = '2023-05-22/50_epochs_UTKFace/'
model.load_for_eval(checkpoint_dir)


app = Flask(__name__)
config_app(app, "flask_config.ini")

db = init_db(app)
jwt = JWTManager(app)
migrate = Migrate(app, db)
CORS(app)
api = Api(app)

image_holder = ImageHolder()

api.add_resource(UploadEndpoint, "/upload", resource_class_kwargs={
    'image_holder': image_holder
})

api.add_resource(GetUploadedEndpoint, "/getUploadedImage", resource_class_kwargs={
    'image_holder': image_holder
})

api.add_resource(GenerateSingleResultEndpoint, '/generateSingleResult', resource_class_kwargs={
    'image_holder': image_holder,
    'model': model
})

api.add_resource(CropFaceEndpoint, '/crop', resource_class_kwargs={
    'image_holder': image_holder
})

api.add_resource(GenerateEvolutionEndpoint, '/generateEvolution', resource_class_kwargs={
    'image_holder': image_holder,
    'model': model
})

api.add_resource(LoginEndpoint, '/auth/login', resource_class_kwargs={
    'database': db
})

api.add_resource(RefreshTokenEndpoint, '/auth/refresh')

api.add_resource(RegisterEndpoint, '/auth/register', resource_class_kwargs={
    'database': db
})

if __name__ == '__main__':
    app.run(debug=True)
