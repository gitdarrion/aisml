#!/usr/bin/env python
# coding: utf-8

# In[5518]:


AB_lengths = []
debug = False

import numpy as np


# In[5520]:


import math


# In[5521]:


def normalize(data): 
    min_data = min(data) 
    max_data = max(data)
    data = np.array([( (d-min_data) / (max_data-min_data) ) for d in data])
    return data


# In[5522]:


def normalize_2D(matrix): 
    min_matrix = matrix.min() 
    max_matrix = matrix.max()
    matrix = (matrix-min_matrix) / (max_matrix-min_matrix)
    return matrix


from sklearn.datasets.samples_generator import make_blobs
AG, y = make_blobs(n_samples=50, centers=5, n_features=2, random_state=0)
AB = np.random.rand(10,2)*10

haf_cell_index=0

def main(AG,AB,iterations): 
    i=0
    while (i<iterations):

        p_select, n_clones, learn_rate, select_percent_of_haf_cells, M_threshold = 0.9, 10, 0.5, 1, 0.8

        # Set of cell affinities. 
        D = [[np.linalg.norm(ag-ab) for ab in AB] for ag in AG] 

        # Arrayify. 
        D=np.array(D) 

        # Set of high affinity (low Euclidean distance) cells.
        H = [list(AB[d[haf_cell_index]]) for d in D.argsort()]

        # Get affinity values for high affinity cells. 
        HD = [D[i][d[haf_cell_index]] for (i,d) in enumerate(D.argsort())]

        # Couple the high affinity cells with their affinity values. 
        Hab = [(a,b) for a,b in zip(H,HD)]

        # Sort the cell-affinity pairs from least to greatest.
        Hab = sorted(Hab, key=lambda k:k[1])

        # Calculate the number of high affinity cells to select based on percentage. 
        select = math.ceil(len(Hab)*p_select)

        # Select the high affinity cells. 
        Hab = Hab[:select]

        # Create sets D and H to hold the affinity values and high affinity cells, respectively. 
        D = [hab[1] for hab in Hab]
        H = [hab[0] for hab in Hab] 
        
        # Normalize the affinity values to decrease the effects of variance.
        D_array = np.array(D)
        D_norm = (D_array-D_array.min()) / (D_array.max() - D_array.min())

        # Increase the value of closer data points - so that they are cloned higher.
        D = np.array([1])-D_norm if haf_cell_index == 0 else D_norm

        # Calculate number of clones. 
        N = D*n_clones

        # Number of clones from floats to integers. 
        N=N.astype(int)

        # Clone the cells. 
        H = np.repeat(H,N,axis=0)

        # Mutate the clones.
        H = np.array([h-learn_rate*(h-AG) for h in H])

        if debug:
            print ("H (Mutated): ")
            print (H)

        # Reduce the dimensionality from 3D to 2D (results from broadcasting the (1,n) dimensional cells to (m,n) dimensions for subtraction)
        H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]))

        if debug:
            print ("H: 3D->2D")
            print (H) 

        # Calculate the affinity values of the mutated cells. 
        D = np.array([[np.linalg.norm(ag-ab) for ab in H] for ag in AG])

        if debug:
            print ("D:")
            print (D)
        
        # Store the affinity values of the highest affinity mutated cells. 
        HD = [D[i][d[haf_cell_index]] for (i,d) in enumerate(D.argsort())]

        if debug:
            print ("HD:")
            print (HD)

        # Find and store the highest affinity mutated cells.
        H = [H[d[haf_cell_index]].tolist() for d in D.argsort()]
        H = np.array(H)

        if debug:
            print ("H:")
            print (H)

        # Select a percentage of the highest affinity cells and store in set M as "memory cells".  
        M = H[:int((select_percent_of_haf_cells)*H.shape[0])]

        if debug:
            print ("M:")
            print (M)

        # Convert M to an array for computing similarity values S. 
        M = np.array(M) 

        if debug:
            print ("M:")
            print (M)

        # Calculate the similarity between memory cells (self-detection). 
        S = np.array([[np.linalg.norm(np.array(mx)-np.array(my)) for mx in M] for my in M])

        if debug:
            print ("S:")
            print (S)

        # Normalize to reduce the effects of variance. 
        S = normalize_2D(S)

        if debug:
            print (S)

        # Remove the lower half of the triangular matrix to stop the double selection of high affinity memory cells. 
        S = [[S[i][j] if j >= i+1 else 0 for j in range(S.shape[1])] for i in range(S.shape[0]-1)]

        # Convert S to an array to use to find and match similarity values with corresponding memory cells.
        S = np.array(S)
        
        # Memory cells with self-similarity values greater than the threshold are selected - because they detect antigens instead of themselves.
        Mp = M[np.argwhere(S>M_threshold)[:,1]]

        # Add the memory cells to the full immune network.
        AB = np.append(AB,Mp,axis=0)

        if debug:
            print (AB.shape) 

        i+=1
    return AG,AB 

AG, AB = main(AG, AB, 1)
print (AB)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.scatter(AB[:,0], AB[:,1])
plt.savefig("AB.png")
plt.clf()

plt.scatter(AG[:,0], AG[:,1])
plt.savefig("AG.png")
plt.clf()