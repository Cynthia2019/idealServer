"""
Nearest neighbors
Author(s): Wei Chen (wchen459@gmail.com)
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
plt.rcParams.update({'font.size': 18})
import seaborn as sns
    

def nn(properties, query_idx, n_neighbors=5):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(properties)
    queried_prop = np.array(properties[query_idx], ndmin=2)
    distances, neighbor_indices = nbrs.kneighbors(queried_prop)
    return neighbor_indices.flatten()


if __name__ == "__main__":
    
    data = pd.read_csv('server/data/freeform_2d.csv')
    properties = np.array(data[['C11', 'C12', 'C22', 'C16', 'C26', 'C66']])
    query_idx = np.random.choice(data.shape[0]) # randomly pick a data point
    print(properties[query_idx])
    # n_neighbors = 5
    # neighbor_indices = nn(properties, query_idx, n_neighbors)
    
    # # Plot
    # from visual2D import visual2D
    # col = n_neighbors + 1
    # plt.figure(figsize=(col*5, 3*5))
    # shape = np.array(data[['unit_cell_x_pixels','unit_cell_y_pixels']])[0]
    # queried_design = list(list(data['geometry_full'])[query_idx])
    # queried_design = np.array(queried_design, dtype=int).reshape(shape)
    # plt.subplot(3, col, 1)
    # plt.imshow(queried_design, cmap='binary')
    # plt.axis('off')
    # plt.title('Queried Design')
    # queried_prop = properties[query_idx]
    # queried_a, queried_E, queried_P = visual2D(queried_prop)
    # plt.subplot(3, col, col+1, polar=True)
    # plt.plot(queried_a, queried_E, '-', lw=3)
    # plt.title("Young's Modulus")
    # plt.subplot(3, col, 2*col+1, polar=True)
    # plt.plot(queried_a, queried_P, '-', lw=3)
    # plt.title("Poisson's Ratio")
    # neighbor_designs = []
    # neighbor_props = []
    # for i, neighbor_idx in enumerate(neighbor_indices):
    #     design = list(list(data['geometry_full'])[neighbor_idx])
    #     design = np.array(design, dtype=int).reshape(shape)
    #     plt.subplot(3, col, i+2)
    #     plt.imshow(design, cmap='binary')
    #     plt.axis('off')
    #     plt.title('Neighbor {}'.format(i+1))
    #     prop = properties[neighbor_idx]
    #     a, E, P = visual2D(prop)
    #     plt.subplot(3, col, col+i+2, polar=True)
    #     plt.plot(a, E, '-', lw=3)
    #     plt.subplot(3, col, 2*col+i+2, polar=True)
    #     plt.plot(a, P, '-', lw=3)
    # plt.tight_layout()
    # plt.show()
    
    