from flask import Flask
app = Flask(__name__)
#Configuration of application, see config.py
#app.config.from_object('config')
#from flask.ext.sqlalchemy import SQLAlchemy
#db = SQLAlchemy(app)
from app import views
