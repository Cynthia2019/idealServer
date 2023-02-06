from django.shortcuts import render
from .apps import ServerConfig

# Create your views here.
# endpoint calls like GET and POST get to directed here 
from django.http import HttpResponse, JsonResponse 
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status 
import os 
import pickle


import numpy as np 

class call_model(APIView): 
    
    def get(self, request): 
        if request.method == 'GET': 
            data = request.GET.get('data').strip('][').split(',')
            name = request.GET.get('name')
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model/model_' + name)
            knn = pickle.load(open(path, 'rb'))
            distances, neighbor_indices = knn.kneighbors(np.array(data, ndmin=2))

            return JsonResponse(neighbor_indices.tolist()[0], safe=False)
