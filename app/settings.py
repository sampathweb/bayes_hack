import os


class Config(object):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class HackConfig(Config):
    APP_SERVER = 'HACK'
    SECRET_KEY = 'HACK - tis is secret?'
