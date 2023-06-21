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
        try:
            '''
            sample body json
            [{"data":[1201642198,258893395.1,891805599.5,0,0,258912099.1]},{"data":[1147656102,199571902.2,972993379.2,0,0,269277239]},{"data":[595738077.6,235145267.1,774648405.4,0,0,117747154.2]},{"data":[1250085484,245054769,957996679.1,0,0,241945479.2]},{"data":[938557646.1,246999636.6,1612174469,0,0,288194472.2]},{"data":[943101313,321845672.3,1515830451,0,0,355549133.2]},{"data":[1303665284,290777122.6,1184266900,0,0,284480219.7]},{"data":[845160247.8,327603175.5,672204343.3,0,0,268449798.5]},{"data":[1118913603,261397156,1153848374,0,0,222020613.7]},{"data":[1322394029,445641277.6,1768747313,0,0,473161355.1]},{"data":[1244002701,309422466.3,1274823221,0,0,306098774.1]},{"data":[938226912.3,313486357.4,1074295017,0,0,305537559.9]},{"data":[1004558727,322039186.2,1675919237,0,0,313860011.7]},{"data":[711837874.6,197359813,1190051154,0,0,179304346.7]},{"data":[649361685,262847658.1,492552757.2,0,0,173078210.1]},{"data":[1022859448,202548204.8,533463100.5,0,0,107653942.9]},{"data":[650700420,490204249,1011705374,0,0,407572128.1]},{"data":[960234952.6,213241847,1010317034,0,0,118363136.2]},{"data":[1291427249,346589739.9,1186878139,0,0,351814463.7]},{"data":[1231281314,408750370.3,1688091340,0,0,425368428.8]},{"data":[1037617612,270599713.6,1375565550,0,0,221185333.2]},{"data":[636133013.1,284772726.9,1125205928,0,0,260805339]},{"data":[754911914.7,207914451.1,1102252332,0,0,212737660]},{"data":[948680275.8,271982077.7,1243946601,0,0,257025710.9]},{"data":[1146436038,286536314.6,1044858940,0,0,314897614.7]},{"data":[804562036.6,237263260.2,1502474650,0,0,213208906.9]},{"data":[1116696358,365392928.6,1437587751,0,0,412996760.9]},{"data":[600993357.3,207463048,1134804686,0,0,186428546.1]},{"data":[1243083795,399428021.8,1638506922,0,0,413015449]},{"data":[1081243679,263575846,1034095494,0,0,243481050.6]},{"data":[631073135.5,206666823.2,1420726773,0,0,200835872.4]},{"data":[1003756863,234443875.8,1260475006,0,0,198049508]},{"data":[1344701404,424219784,1910864826,0,0,482590081]},{"data":[808007596.3,194431850.5,933021381.5,0,0,206500030]},{"data":[1009978317,239269113.1,1009978317,0,0,108583866.7]},{"data":[1133092736,297005696.2,1133092736,0,0,147873206.3]},{"data":[1263422928,363416622.9,1263422928,0,0,194629201.4]},{"data":[608069835.3,233978482,608069835.3,0,0,178115486.7]},{"data":[608069835.3,233978482,608069835.3,0,0,178115486.7]},{"data":[810565198.9,323413695.8,810565198.9,0,0,242160288.9]},{"data":[810565198.9,323413695.8,810565198.9,0,0,242160288.9]},{"data":[1040780503,424446567.6,1040780503,0,0,314940502.1]},{"data":[1040780503,424446567.6,1040780503,0,0,314940502.1]},{"data":[1304213019,540295724,1304213019,0,0,395789611.9]},{"data":[902722050.9,193182656.6,902722050.9,0,0,66821719.89]},{"data":[902722050.9,193182656.6,902722050.9,0,0,66821719.89]},{"data":[1145788681,303214818,1145788681,0,0,138636386.2]},{"data":[1281767701,371600373.7,1281767701,0,0,177849815.5]},{"data":[1216962993,229857344.1,807416334.9,0,0,79411954.89]},{"data":[1216962993,229857344.1,807416334.9,0,0,79411954.89]},{"data":[807416334.9,229857344.1,1216962993,0,0,79411954.89]},{"data":[807416334.9,229857344.1,1216962993,0,0,79411954.89]},{"data":[935100210.1,310991423.4,1455875777,0,0,128009210.4]},{"data":[1058736321,378154087.2,1598611671,0,0,167816449.7]},{"data":[1203911967,482249429.5,1853657268,0,0,242391995.8]},{"data":[523509483.3,412599797.4,510593773.5,0,0,316530250.8]},{"data":[627951801,465339228.4,611097582.2,0,0,351028801.9]},{"data":[745831205.1,517123711.2,724494257.9,0,0,384571091.3]},{"data":[878442680.4,568363987.3,852178085.9,0,0,417248602.4]},{"data":[1026599064,620089030.8,995165752,0,0,449192874.3]},{"data":[668021306.4,196011960.4,1207145588,0,0,198879529.7]}]
            '''
            body = json.loads(request.body)
            prop = []
            for d in body: 
                for key, value in d.items(): 
                    prop.append(value)
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

            response = JsonResponse([{
                'raw': np.round(mean_dist_raw, 3), 
                'standardized': np.round(mean_dist_standardized, 3)
            }], status=200, safe=False)
            response['Access-Control-Allow-Origin'] = '*'
            return response

        except Exception as e:
            res = JsonResponse({
                'error': str(e)
            }, status=500)
            res['Access-Control-Allow-Origin'] = '*'
            return res
