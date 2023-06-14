import configparser
from datetime import timedelta


def read_db_credentials(file_path):
    config_string = 'DATABASE'
    config = configparser.ConfigParser()
    config.read(file_path)
    database_credentials = {}
    for key in config[config_string]:
        database_credentials[key] = config[config_string][key]

    return database_credentials


def read_jwt_attributes(file_path):
    config_string = 'JWT'
    config = configparser.ConfigParser()
    config.read(file_path)
    jwt_attributes = {}
    for key in config[config_string]:
        jwt_attributes[key] = config[config_string][key]

    return jwt_attributes


def load_model(model, file_path):
    config_string = 'MODEL'
    config = configparser.ConfigParser()
    config.read(file_path)
    model_attributes = {}
    for key in config[config_string]:
        model_attributes[key] = config[config_string][key]

    model.load_for_eval(model_attributes['path'])


def parse_date(date: str):
    date_parts = date.split(' ')

    time_units = {
        'seconds': 0,
        'minutes': 0,
        'hours': 0,
        'days': 0
    }
    for part in date_parts:
        if part[-1] == 's':
            time_units['seconds'] = int(part[0:-1])
        elif part[-1] == 'm':
            time_units['minutes'] = int(part[0:-1])
        elif part[-1] == 'h':
            time_units['hours'] = int(part[0:-1])
        elif part[-1] == 'd':
            time_units['days'] = int(part[0:-1])

    return time_units


def config_app(app, file_path):
    db_credentials = read_db_credentials(file_path)
    jwt_attributes = read_jwt_attributes(file_path)

    app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://" + db_credentials['username'] + ":" + \
                                            db_credentials['password'] + "@" + db_credentials['host'] + ":" + \
                                            db_credentials['port'] + '/' + \
                                            db_credentials['db']

    app.config['JWT_SECRET_KEY'] = jwt_attributes['jwt_secret_key']
    access_token_time_units = parse_date(jwt_attributes['jwt_access_token_expires'])
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=access_token_time_units['days'],
                                                       hours=access_token_time_units['hours'],
                                                       minutes=access_token_time_units['minutes'],
                                                       seconds=access_token_time_units['seconds'])
    refresh_token_time_units = parse_date(jwt_attributes['jwt_refresh_token_expires'])
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=refresh_token_time_units['days'],
                                                        hours=refresh_token_time_units['hours'],
                                                        minutes=refresh_token_time_units['minutes'],
                                                        seconds=refresh_token_time_units['seconds'])
    app.config['JWT_TOKEN_LOCATION'] = jwt_attributes['jwt_token_location']