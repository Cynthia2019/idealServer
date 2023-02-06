import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle

def save_model(path, filename, n_neighbors=5): 
    data = pd.read_csv(path + filename)
    properties = np.array(data[['C11', 'C12', 'C22', 'C16', 'C26', 'C66']])
    n_neighbors = 5
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(properties)

    #save to bin 
    knnModel = open('model/model_' + filename, 'wb')
    pickle.dump(knn, knnModel)

    knnModel.close()

if __name__=="__main__":
    path = 'data/'
  #  filename = 'lattice_2d_sample.csv'
  #  filename = 'freeform_2d.csv'
    filename = 'freeform_2d_sample.csv'
    save_model(path, filename)