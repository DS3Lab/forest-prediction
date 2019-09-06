import numpy as np
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import glob
import os

img_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped/train'
img_paths = glob.glob(os.path.join(img_dir, '*'))

X_train = []

i = 0
for path in img_paths:
    if i%1000==0:
        print(i)
    img_arr = cv2.imread(path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    X_train.append(img_arr)
    i = i + 1

X_train = np.array(X_train)
print('Original data shape,', X_train.shape)
X_train = X_train.reshape(X_train.shape[0], -1)
print('Data reshape,', X_train.shape)

print('Normalizing data...')
X_train_norm = X_train / 255.
print('Double check min and max', X_train_norm.min(), X_train_norm.max())
train_mean = np.mean(X_train_norm, axis=0)
train_std = np.std(X_train_norm, axis=0)
print('Mean, Std', train_mean, train_std)

X_train_norm = X_train_norm - train_mean

print('Calculating covariance matrix...')
cov = np.dot(X_train_norm.T, X_train_norm) / X_train_norm.shape[0]
print('Covariance shape', cov.shape)

print('Calculating SVD...')
U, S, V = np.linalg.svd(cov)

print('Saving SVD...')
svd = {
    'U': U,
    'S': S,
    'V': V
}
with open('svd.pkl', 'wb') as f:
    pkl.dump(svd, f, protocol=pkl.HIGHEST_PROTOCOL)

print('Calculating white matrix...')
epsilon = 0.1
sqlam = np.sqrt(S+epsilon)
zcaWhiteMat = np.dot(U/sqlam[np.newaxis , :],U.T)

print('Saving white matrix...')
with open('white_matrix.pkl', 'wb') as f:
    pkl.dump(zcaWhiteMat, f, protocol=pkl.HIGHEST_PROTOCOL)


# trainX = np.dot(trainX,zcaWhiteMat.T)
# print ('TrainX ZCA Done')
# validX = np.dot(validX,zcaWhiteMat.T)
# print ('ValidX ZCA Done')
# testX = np.dot(testX,zcaWhiteMat.T)
# print ('TestX ZCA Done')
