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
            prop = []
            for d in body: 
                for key, value in d.items(): 
                    prop.append(value)
            # data = request.GET.get('data').strip('][').split(',')
            prop = np.array(prop)
            N = prop.shape[0]
            M = prop.shape[1]
            zero_col = np.all(prop[..., :]==0, axis=0) 
            prop = prop[:, ~zero_col].reshape(N, M-2)
            prop_raw = prop
            prop = prop_raw
            ## data normalization
            n_data = len(prop_raw)
            n_threshold = 1000
            if n_data <= n_threshold:
                deno = pdist(prop).mean()
            else:
                tmp = np.arange(n_data)
                np.random.shuffle(tmp)
                idx_rnd = tmp[:n_threshold]
                deno = pdist(prop[idx_rnd]).mean()
            prop = prop_raw / deno
            def standardize(data):  # component-wise standardization
                return (data-np.mean(data, axis=0))/np.std(data, axis=0) 
            prop = standardize(prop)


            ## property diversity
            idx_left = np.arange(len(prop))
            D_feature = 3000  # for N(ground set) >= 10k, approximation must be used to avoid too much overhead 
            batch_size = 20   # for N(subset) >= 100, it's recommended to iterate with a small batch size (e.g., 200 = 20 iter x 10)
            if n_data <= 3000: # greedy algorithm
                C = decompose_kernel(squareform(pdist(prop)))
                idx_abs = sample_k(C['D'], batch_size)
            else: # approximated algorithm (based on Fourier feature)
                Vp = RFF(prop, D_feature) 
                _, idx_abs, _ = k_Markov_dual_DPP(Vp, idx_left, batch_size=batch_size)

            ## diversity of the subset (mean Euclidean distance)
            mean_dist_standardized = np.mean(pdist(prop[idx_abs])) # mean Euclidean distance of standardized property
            mean_dist_raw = np.mean(pdist(prop_raw[idx_abs]))      # mean Euclidean distance of raw property
            print(mean_dist_raw, mean_dist_standardized)
            return JsonResponse([{
                'raw': np.round(mean_dist_raw, 3), 
                'standardized': np.round(mean_dist_standardized, 3)
            }], safe=False)
