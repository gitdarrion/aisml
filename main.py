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



haf_cell_index=0

def main(AG, AB, iterations, p_select, n_clones, learn_rate, p_haf_cells, M_threshold): 
    i=0

    print ("AB Shape:")
    print (AB.shape)

    while (i<iterations):


        # Set of cell affinities. 
        D = [[np.linalg.norm(ag-ab) for ab in AB] for ag in AG] 

        # Arrayify. 
        D = np.array(D) 

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
        D = np.repeat(D,N,axis=0)

        if debug:
            print ("H (Pre-Mutated): ")
            print (H.shape)

        # Mutate the clones.
        # Learn rate d is proportional to distance.
        H = np.array([h-(1-d)*(h-AG) for h,d in zip(H,D)])

        if debug:
            print ("H (Mutated): ")
            print (H.shape)

        # Reduce the dimensionality from 3D to 2D (results from broadcasting the (1,n) dimensional cells to (m,n) dimensions for subtraction)
        H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]))

        if debug:
            print ("H: 3D->2D")
            print (H.shape) 

        # Calculate the affinity values of the mutated cells. 
        D = np.array([[np.linalg.norm(ag-ab) for ab in H] for ag in AG])

        if debug:
            print ("D before argsort:")
            print (D.shape)
        
        # Store the affinity values of the highest affinity mutated cells. 
        HD = [D[i][d[haf_cell_index]] for (i,d) in enumerate(D.argsort())]

        if debug:
            print ("HD after argsort:")
            print (np.array(HD).shape)

        # Find and store the highest affinity mutated cells.
        H = [H[d[haf_cell_index]].tolist() for d in D.argsort()]
        H = np.array(H)

        if debug:
            print ("H:")
            print (H.shape)

        # Select a percentage of the highest affinity cells and store in set M as "memory cells".  
        M = H[:int((p_haf_cells)*H.shape[0])]

        # Convert M to an array for computing similarity values S. 
        M = np.array(M) 

        if debug:
            print ("M:")
            print (M.shape)

        # Calculate the similarity between memory cells (self-detection). 
        S = np.array([[np.linalg.norm(np.array(mx)-np.array(my)) for mx in M] for my in M])

        if debug:
            print ("S:")
            print (S.shape)

        # Normalize to reduce the effects of variance. 
        S = normalize_2D(S)

        if debug:
            print (S.shape)

        # Remove the lower half of the triangular matrix to stop the double selection of high affinity memory cells. 
        #S = [[S[i][j] if j >= i+1 else 0 for j in range(S.shape[1])] for i in range(S.shape[0]-1)]

        # Convert S to an array to use to find and match similarity values with corresponding memory cells.
        S = np.array(S)
        
        # Clonal suppression. 
        # Memory cells with self-similarity values greater than the threshold are selected - because they detect antigens instead of themselves.
        Mp = M[list(set(np.argwhere(S>M_threshold)[:,1]))]

        # Add the memory cells to the full immune network.
        AB = np.append(AB,Mp,axis=0)

        # Network suppression. 
        S = np.array([[np.linalg.norm(np.array(abx)-np.array(aby)) for abx in AB] for aby in AB])

        # Remove the lower half of the triangular matrix. 
        #S = [[S[i][j] if j >= i+1 else 0 for j in range(S.shape[1])] for i in range(S.shape[0]-1)]

        S = np.array(S) 

        if debug:
            print ("S0:")
            print (S.shape)

        AB = AB[list(set(np.argwhere(S>M_threshold)[:,1]))]

        print (AB.shape) 

        i+=1
    return AB 

AG = np.random.rand(100,2)*10
print (AG.shape)
# p_select, n_clones, learn_rate, p_haf_cells, M_threshold
# 0.9, 10, 0.5, 1, 0.8

""" 
    With the number of antibodies less than antigens (by a factor of 10), then clustering increases in accuracy. 
"""

def execute(AG=AG, iterations=10, p_select=0.2, n_clones=10, learn_rate=0.5, p_haf_cells=1, M_threshold=0.8, n_experiment=1):
    

    AB = np.random.rand(10,2)*10
    AB = main(AG, AB, iterations=iterations, p_select=p_select, n_clones=n_clones, learn_rate=learn_rate, p_haf_cells=p_haf_cells, M_threshold=M_threshold)
    
    #print (AB.shape)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    title = 0.1+(0.1*n_experiment)
    #plt.title("p_select=" + str(title))
    plt.scatter(AB[:,0], AB[:,1])
    plt.savefig("AB" + str(n_experiment) + ".png")
    plt.clf()

    #plt.title("p_select=" + str(title))
    plt.scatter(AG[:,0], AG[:,1])
    plt.savefig("AG" + str(n_experiment) + ".png")
    plt.clf()

execute(n_experiment=1)
# execute(p_select=0.3, n_experiment=2)
# execute(p_select=0.5, n_experiment=3)
# execute(p_select=0.7, n_experiment=4)
# execute(p_select=0.9, n_experiment=5)
# execute(p_select=1.0, n_experiment=6)