import numpy as np

def normalizeMatrix(matrix, etendue):
    # Normalisation for [0,1]
    matrix_norm = (matrix-np.min(matrix))/np.ptp(matrix)
 
    # Normalisation considering the noise
    matrix_norm_noise = matrix_norm*(1-etendue)+etendue/2
    return matrix_norm_noise