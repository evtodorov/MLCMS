import numpy as np
def pca_image_processing(U, S, VT, no_pca):
    if no_pca == "all":
        S_eye = S * np.eye(U.shape[0], VT.shape[0])
        constr_data = np.matmul(np.matmul(U, S_eye), VT)
    else:
        S_img = np.zeros(S.shape)
        S_img[0:no_pca] = S[0:no_pca]

        S_img_eye = S_img * np.eye(U.shape[0], VT.shape[0])
        constr_data = np.matmul(np.matmul(U, S_img_eye), VT)

    return constr_data