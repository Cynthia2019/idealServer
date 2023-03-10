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
import io 
from pathlib import Path

import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np 

import boto3
from dotenv import load_dotenv

load_dotenv()

colorAssignment = [
  "#FFB347", "#8A8BD0", "#FFC0CB", '#6FA8DC', '#8FCE00', '#CC0000', '#38761D', '#9FC5E8', '#2f3b45','#e8c29f'
]

class call_model(APIView): 
    
    def get(self, request): 
        if request.method == 'GET': 
            data = request.GET.get('data').strip('][').split(',')
            dir = os.path.dirname(os.path.realpath(__file__))
            path = dir + '/model'
            knn = pickle.load(open(path, 'rb'))
            distances, neighbor_indices = knn.kneighbors(np.array(data, ndmin=2, dtype=float))

            return JsonResponse(neighbor_indices.tolist()[0], safe=False)
    
    def post(self, request): 
        if request.method == 'POST': 
            s3 = boto3.client('s3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
            )
            names = []
            # fetch all files in s3 
            total_df = pd.DataFrame()
            response = s3.list_objects(
                Bucket='ideal-dataset-1', 
            )['Contents']
            for content in response: 
                names.append(content['Key'])

            print(names)

            for index, name in enumerate(names): 
                response = s3.get_object(
                    Bucket='ideal-dataset-1', 
                    Key=name
                )['Body']
                df = pd.read_csv(io.BytesIO(response.read()))
                df['dataset_name'] = name
                df['dataset_color'] = colorAssignment[index]
                total_df = pd.concat([total_df, df], ignore_index=True)
            curr_path = os.path.realpath(os.path.dirname(__file__))

            
            # merge the data into one big df
            # save the final model 
            def save_model(data, curr_path, n_neighbors=5): 
                n_neighbors = 5
                knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data[['C11', 'C12', 'C22', 'C16', 'C26', 'C66']].values)
                #save to bin 
                knnModel = open(curr_path+'/model', 'wb')
                pickle.dump(knn, knnModel)

                knnModel.close()
            save_model(total_df, curr_path, 5)

            total_df.to_csv("total_df.csv")

            return JsonResponse(total_df.to_json(orient="records"), safe=False)
