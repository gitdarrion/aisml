import numpy as np

class ImmuneNetwork:

    def __init__(self):
        return None 

    def fit(
            self,                                               \
            Data:               np.ndarray=None,                \
            Detectors:          np.ndarray=None,                \
            DetectionRatesXY:   np.ndarray=None,                \
            DetectionRatesYY:   np.ndarray=None,                \
            n_clones:           int=0,                          \
            n_maximums:         int=0,                          \
            learning_rate:      int=0,                          \
            clone_rate:         int=0,                          \
            affinity_rate:      int=0,                          \
            deletion_rate:      int=0,                          \
            suppression_rate:   int=0                           \
            ):

        for i in range(len(Data)):
            
            """
                X:  Matrix of data vectors. 
                Y:  Matrix of detector vectors. 
                D:  Matrix of Euclidean distances between x and y.
                ix: List of indices.
                Z:  Matrix of detector vector clones.

            """

            X,Y = [np.array([Data[i]]*len(Detectors)), Detectors] 
            D = np.array([np.linalg.norm(x-y) for x,y in zip(X,Y)]) 
            ix = D.argsort()[-1*n_maximums:][::-1]         
            D = D[ix]
            D = D/D.sum()
            D = D*n_clones
            D = D.astype(int) 
            Z = Y[ix]
            Z = np.repeat(Z,D,axis=0)
            if i>0: 
                C = C-learning_rate*(C-Z) 
            
            """
                Y = Y - learning_rate * (Y[DRXY_ix]-X)
                DRXY = [np.linalg.norm(x-y) for x,y in zip(X,Y)]
                DRXY = np.array(DRXY)
            """

# delta = number of highest affinity cells. 
# n_clones = total number of clones.
im = ImmuneNetwork()
im.fit(np.random.randint(10,size=(1,5)), np.random.randint(10,size=(10,5)), n_clones=100, n_maximums=10, learning_rate=0.2)