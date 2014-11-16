import os


class Config(object):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class HackConfig(Config):
    DEBUG = True
    APP_SERVER = 'HACK'
    SECRET_KEY = 'HACK - tis is secret?'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:bayeshack@bayesimpact.soumet.com/bayes'
