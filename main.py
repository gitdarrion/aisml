import numpy as np

class ImmuneNetwork:

    def __init__(self):
        return None 

    """
    PrimeDetectors, DetectionRatesYY, DetectionRatesXY, clone_rate, affinity_rate, deletion_rate, suppression_rate
    """
    def fit(self, Data, Detectors, n, maximums, learning_rate):

        for i in range(len(Data)):

            X = [Data[i]]*len(Detectors)
            X = np.array(X)
            Y = Detectors 
            DRXY = [np.linalg.norm(x-y) for x,y in zip(X,Y)]
            DRXY = np.array(DRXY)
            DRXY_ix = DRXY.argsort()[-1*maximums:][::-1]
            DRXY = DRXY[DRXY_ix]
            DRXY = DRXY/DRXY.sum()
            DRXY_clones = DRXY*n 
            DRXY_clones = DRXY_clones.astype(int)
            Y_clones = Y[DRXY_ix]
            Y_clones = [[Y_clones[ix]]*DRXY_clones[ix] for ix in DRXY_ix]
            Y_clones = np.array(Y_clones)
            Y = Y - learning_rate * (Y[DRXY_ix]-X)
            DRXY = [np.linalg.norm(x-y) for x,y in zip(X,Y)]
            DRXY = np.array(DRXY)


# delta = number of highest affinity cells. 
# N = total number of clones.
im = ImmuneNetwork()
im.fit(np.random.randint(10,size=(1,5)), np.random.randint(10,size=(10,5)), n=100, maximums=10, learning_rate=0.2)