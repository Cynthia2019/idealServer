from django.apps import AppConfig
from sklearn.neighbors import NearestNeighbors
import pickle
import os 

class ServerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "server"