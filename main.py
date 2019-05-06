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

            X = [Data[i]]*len(Detectors)
            X = np.array(X)
            Y = Detectors 
            DRXY = [np.linalg.norm(x-y) for x,y in zip(X,Y)]
            DRXY = np.array(DRXY)
            DRXY_ix = DRXY.argsort()[-1*n_maximums:][::-1]
            DRXY = DRXY[DRXY_ix]
            DRXY = DRXY/DRXY.sum()
            DRXY_clones = DRXY*n_clones 
            DRXY_clones = DRXY_clones.astype(int)
            Y_clones = Y[DRXY_ix]
            Y_clones = [[Y_clones[ix]]*DRXY_clones[ix] for ix in DRXY_ix]
            Y_clones = np.array(Y_clones)
            Y = Y - learning_rate * (Y[DRXY_ix]-X)
            DRXY = [np.linalg.norm(x-y) for x,y in zip(X,Y)]
            DRXY = np.array(DRXY)


# delta = number of highest affinity cells. 
# n_clones = total number of clones.
im = ImmuneNetwork()
im.fit(np.random.randint(10,size=(1,5)), np.random.randint(10,size=(10,5)), n_clones=100, n_maximums=10, learning_rate=0.2)