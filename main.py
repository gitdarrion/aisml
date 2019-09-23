import numpy as np

class ImmuneNetwork:

    def __init__(X,C,D,S,a,b,c,d,e,f):
        self.X = X        # Data Matrix 
        self.C = C        # Network Cell Matrix 
        self.D = D        # Dissimilarity Matrix 
        self.S = S        # Similarity Matrix 
        self.a = a        # Number of Clones 
        self.b = b        # Number of High Affinity Cells to Clone and Mutate
        self.c = c        # Percentage of Mature Cells to be Selected 
        self.d = d        # Death Rate 
        self.e = e        # Suppression Rate 
        self.f = f        # Number of Iterations
        return None 

    def fit(): 
        for iteration in range(self.f): 
            for i in range(len(self.X)):
                for j in range(len(self.C)): 
                    D[i][j] = np.linalg.norm(self.X[i]-self.C[j])
                    
        return None 

    def predict(): 
        return None