from django.shortcuts import render
from ..apps import ServerConfig

# Create your views here.
# endpoint calls like GET and POST get to directed here 
from django.http import HttpResponse, JsonResponse 
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status 
import os 
import pickle
import json
from ..scripts.DPPlibrary import * 

import numpy as np 

class call_diversity(APIView): 
    
    def post(self, request): 
        if request.method == 'POST': 
            # data would be sth like [1.37580596e+00 5.88081175e-01 1.37995905e+00 5.13899350e-04 6.52366007e-04 4.27444853e-01]
            #[2250985995,686328363.6,1508385126,0,0,393330312.6]
            body = json.loads(request.body)
            data = []
            for d in body: 
                for key, value in d.items(): 
                    data.append(value)
            # data = request.GET.get('data').strip('][').split(',')
            data = np.array(data)
            data_raw = data
            data = (data-np.mean(data, axis=0))/np.std(data, axis=0) if np.std(data, axis=0).all() else (data-np.mean(data, axis=0))
            idx_left = np.arange(len(data))
            D_feature = 3000 
            batch_size = 20 
            Vp = RFF(data, D_feature) 
            idx_rel, idx_abs, _ = k_Markov_dual_DPP(Vp, idx_left, batch_size=batch_size)

            ## diversity of the subset (mean Euclidean distance)
            mean_dist_standardized = np.mean(pdist(data[idx_abs])) # mean Euclidean distance of standardized property
            mean_dist_raw = np.mean(pdist(data_raw[idx_abs]))      # mean Euclidean distance of raw property


            return JsonResponse(np.round(mean_dist_raw, 3))

    def standardize(data):  # auxiliary function 
        return (data-np.mean(data, axis=0))/np.std(data, axis=0) 
