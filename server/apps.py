from django.apps import AppConfig
from sklearn.neighbors import NearestNeighbors
import pickle
import os 

class ServerConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "server"
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model_freeform_2d')
    # knn = pickle.load(open(path, 'rb'))