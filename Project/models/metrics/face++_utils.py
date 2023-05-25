import json

from facepplib import *
import configparser


def read_facepp_credentials(file_path):
    config_string = 'KEYS'
    config = configparser.ConfigParser()
    config.read(file_path)
    facepp_keys = {}
    for key in config[config_string]:
        facepp_keys[key] = config[config_string][key]

    return facepp_keys


def read_caae_paths(file_path):
    config_string = 'CAAE'
    config = configparser.ConfigParser()
    config.read(file_path)
    caae_path_keys = {}
    for key in config[config_string]:
        caae_path_keys[key] = config[config_string][key]

    return caae_path_keys


def create_facepp_client(file_path):
    facepp_keys = read_facepp_credentials(file_path)
    return FacePP(api_key=facepp_keys['api_key'], api_secret=facepp_keys['api_secret'])