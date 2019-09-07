import numpy as np
import pickle as pkl
import cv2
import matplotlib.pyplot as plt
import glob
import os

def cal_white_matrix():
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

def get_white_matrix():
    white_matrix_path = '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/scripts/white_matrix.pkl'
    if not os.path.exists(white_matrix_path):
        cal_white_matrix()
    with open(white_matrix_path, 'rb') as f:
        white_matrix = pkl.load(f)

def get_data(img_dir):
    pass

def main():
    partition_names = ['train', 'val', 'test']
    quad_list = [get_imgs(os.path.join(args.input_dir, 'train')),
        get_imgs(os.path.join(args.input_dir, 'val')),
        get_imgs(os.path.join(args.input_dir, 'test')),
    ]
    X_train, X_val, X_test = get_data()
    zca_white_mat = get_white_matrix()
    X_train_whiten =np.dot(X_train, zca_white_mat.T)
    X_val_whiten =np.dot(X_val, zca_white_mat.T)
    X_test_whiten =np.dot(X_test, zca_white_mat.T)

if __name__ == '__main__':
    main()

# trainX = np.dot(trainX,zcaWhiteMat.T)
# print ('TrainX ZCA Done')
# validX = np.dot(validX,zcaWhiteMat.T)
# print ('ValidX ZCA Done')
# testX = np.dot(testX,zcaWhiteMat.T)
# print ('TestX ZCA Done')
