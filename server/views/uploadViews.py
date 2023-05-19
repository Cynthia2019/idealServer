# from django.shortcuts import render
# from ..apps import ServerConfig

# # Create your views here.
# # endpoint calls like GET and POST get to directed here 
# from django.http import HttpResponse, JsonResponse 
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status 
# import os 
# import pickle
# import json
# from ..scripts.calculate2D import calculate2D
# import concurrent.futures
# import pandas as pd

# import numpy as np 

# class call_upload(APIView): 
    
#     def post(self, request): 
#         if request.method == 'POST': 
#             # data would be sth like [[1.37580596e+00 5.88081175e-01 1.37995905e+00 5.13899350e-04 6.52366007e-04 4.27444853e-01],[2250985995,686328363.6,1508385126,0,0,393330312.6]]
#             body = json.loads(request.body)
#             body = pd.DataFrame(body)
#             data = body.values.tolist()
#             total_

#             with concurrent.futures.ProcessPoolExecutor() as executor:
#                 results = executor.map(calculate2D, data)
#             return JsonResponse([{
#                 'estimation': estimation,
#             }], safe=False)
