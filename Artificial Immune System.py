    #!/usr/bin/env python
    # coding: utf-8

    # In[5518]:


    AB_lengths = []


    # In[5519]:


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
        #matrix = np.array([[( (y-min_data) / (max_data-min_data) ) for y in x] for x in matrix])


    # In[5523]:


from sklearn.datasets.samples_generator import make_blobs


# In[5524]:


AG, y = make_blobs(n_samples=2, centers=3, n_features=2, random_state=0)


# In[5525]:


AB = np.random.rand(2,2)*10


def main(): 

    # In[5526]:


    p_select, n_clones, learn_rate, OMEGA, M_threshold = 0.9, 100, 1, 0.5, 0.9


    # In[5527]:


    


    # In[5528]:


    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    D = [[np.linalg.norm(ag-ab) for ab in AB] for ag in AG]


    D=np.array(D)

    # Select the n highest affinity cells. 
    # The cells with the least distance are first in the argsorted lists. 
    H = [list(AB[d[0]]) for d in D.argsort()]





    # Store the affinity values of the highest affinity cells. 
    # H and HD map directly to one another. 
    # So, the affinity of the first cell in H is the first value in HD. 
    HD = [D[i][d[0]] for (i,d) in enumerate(D.argsort())]




    # Pair the HAF cells with HD affinities for sorting. 
    Hab = [(a,b) for a,b in zip(H,HD)]




    Hab = sorted(Hab, key=lambda k:k[1])


    


    select = math.ceil(len(Hab)*p_select)



    Hab = Hab[:select]


  

    D = [hab[1] for hab in Hab]




    H = [hab[0] for hab in Hab] 



    D_array = np.array(D)


    


    Dnorm = (D_array-D_array.min()) / (D_array.max() - D_array.min())




    D = np.array([1])-Dnorm





    N = D*n_clones


    # In[5558]:


    N=N.astype(int)


    


    H = np.repeat(H,N,axis=0)





    H = np.array([h-learn_rate*(h-AG) for h in H])


    


    H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]))


    

    D = np.array([[np.linalg.norm(ag-ab) for ab in H] for ag in AG])




    D.shape


    # In[5569]:


    HD = [D[i][d[0]] for (i,d) in enumerate(D.argsort())]




    H=[H[d[0]].tolist() for d in D.argsort()]



    Hab = H


    # In[5574]:


    


    # In[5576]:


    M = Hab[:int(OMEGA*len(Hab))]




    M = np.array(M)



    # Clonal suppression. 
    S = np.array([[np.linalg.norm(np.array(mx)-np.array(my)) for mx in M] for my in M])




    S=normalize_2D(S)




    S=[[S[i][j] if j >= i+1 else 0 for j in range(S.shape[1])] for i in range(S.shape[0]-1)]


    # In[5585]:


    S=np.array(S)


    # In[5586]:


    np.argwhere(S>M_threshold)[:,1]


    # In[5587]:


    Mp=M[np.argwhere(S>M_threshold)[:,1]]


    # In[5588]:


    Mp=np.array(list(set([tuple(mp) for mp in Mp])))




    AB=np.append(AB,Mp,axis=0)




    with open("ais.txt", "w+") as of: 
        of.write(np.array2string(AB, separator=',') + "\n")
        of.close()


    # In[5597]:


    print (AB_lengths)


  

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.scatter(AB[:,0], AB[:,1])
    plt.savefig("AB.png")
    plt.show()

    plt.close()

    plt.scatter(AG[:,0], AG[:,1])
    plt.savefig("AG.png")
    plt.show()

    plt.close()




main()

