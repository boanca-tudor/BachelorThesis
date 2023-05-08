from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ApplicationUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String)
    email = db.Column(db.String)
    password = db.Column(db.String)

    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

    def __repr__(self):
        return f"{self.username} {self.email}"


def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
    return db
