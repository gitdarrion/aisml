import numpy as np

def fit(X, C, a, hn, hp, cs, nc):
    
    i=0
    
    while (i<100):
        
        D = np.array([np.linalg.norm(c-X) for c in C])
        ixs = D.argsort()[-hn:][::1]
        HAF_Cells_D = D[ixs]
        HAF_Cells_N = (HAF_Cells_D / sum(HAF_Cells_D)) * nc
        HAF_Cells_N = HAF_Cells_N.astype(int) 
        HAF_Cells = C[ixs]
        HAF_Cells = np.repeat(HAF_Cells, HAF_Cells_N, axis=0)
        HAF_Cells = np.array([cell-a*(cell-X) for cell in HAF_Cells])
        HAF_Cells = np.reshape(HAF_Cells, (HAF_Cells.shape[0]*HAF_Cells.shape[1], HAF_Cells.shape[2]))

        D = np.array([np.linalg.norm(cell-X) for cell in HAF_Cells])
        ixs = D.argsort()[int((1-hp)*D.shape[0]):][::1]
        D = D[ixs]
        M = HAF_Cells[ixs]

        S = np.array([np.linalg.norm(m-M) for m in M])
        ixs = S.argsort()[-int((1-cs)*S.shape[0]):][::1]
        M = M[ixs]
        C = np.append(C, M, axis=0)

        S = np.array([np.linalg.norm(c-C) for c in C])

        ixs = S.argsort()[-int((1-cs)*S.shape[0]):][::1]
        C = C[ixs]
        
        i+=1
        
    print (C.shape)
    print (C)
