# Create your views here.
# endpoint calls like GET and POST get to directed here
from django.http import JsonResponse
from rest_framework.views import APIView

from sklearn.neighbors import NearestNeighbors
import json


class call_model(APIView):
    # get knn results
    # request body has a data df with all the data to be refit
    def post(self, request):
        try:
            '''
            sample body json
            {
                "data": [[1201642198,258893395.1,891805599.5,0,0,258912099.1],
                [1147656102,199571902.2,972993379.2,0,0,269277239],
                [595738077.6,235145267.1,774648405.4,0,0,117747154.2],
                [1250085484,245054769,957996679.1,0,0,241945479.2],
                [938557646.1,246999636.6,1612174469,0,0,288194472.2],
                [943101313,321845672.3,1515830451,0,0,355549133.2],
                [1303665284,290777122.6,1184266900,0,0,284480219.7],
                [845160247.8,327603175.5,672204343.3,0,0,268449798.5],
                [1118913603,261397156,1153848374,0,0,222020613.7],
                [1322394029,445641277.6,1768747313,0,0,473161355.1]],
                "dataPoint":
                    [[1201642198,258893395.1,891805599.5,0,0,258912099.1]]
            }
            '''
            body = json.loads(request.body.decode('utf-8'))
            data = body['data']
            dataPoint = body['dataPoint']
            if dataPoint is None:
                res = JsonResponse({
                    'error': 'data parameter is missing'
                }, status=400)
                res['Access-Control-Allow-Origin'] = '*'
                return res
            n_neighbors = 5

            knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                algorithm='ball_tree').fit(data)
            distances, neighbor_indices = knn.kneighbors(dataPoint)

            response = JsonResponse({
                'distances': distances.tolist()[0],
                'indices': neighbor_indices.tolist()[0]
            }, status=200)
            response['Access-Control-Allow-Origin'] = '*'
            return response

        except Exception as e:
            res = JsonResponse({
                'error': str(e)
            }, status=500)
            res['Access-Control-Allow-Origin'] = '*'
            return res
