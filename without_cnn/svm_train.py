import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Config
IMG_SIZE = (128,128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)
SVM_PARAMS = dict(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

def extract_features(filenames, folder, image_base):
    dummy = np.zeros(IMG_SIZE, dtype=np.float32)
    dummy_hog = hog(dummy, **HOG_PARAMS)
    feats=[]
    for fname in tqdm(filenames, desc=f'Extracting HOG ({folder})'):
        base = fname.split('_',1)[1]
        view_feats=[]
        for p in ['a_','c_','s_']:
            path=os.path.join(image_base, folder, p+base)
            if not os.path.exists(path): view_feats.append(dummy_hog)
            else:
                g=cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                g=cv2.resize(g, (IMG_SIZE[1],IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
                view_feats.append(hog(g, **HOG_PARAMS))
        feats.append(np.hstack(view_feats))
    return np.vstack(feats)

if __name__=='__main__':
    df = pd.read_pickle('/without_cnn/train_clean.pkl')
    X_files = df['id'].values; y = df['class'].values
    X_train_files, X_val_files, y_train, y_val = train_test_split(X_files, y, test_size=0.2, stratify=y, random_state=42)
    X_train = extract_features(X_train_files, 'train', 'organ-all-mnist-DL20242025')
    X_val   = extract_features(X_val_files,   'train', 'organ-all-mnist-DL20242025')
    scaler = StandardScaler(); X_train_s = scaler.fit_transform(X_train); X_val_s = scaler.transform(X_val)
    clf = SVC(**SVM_PARAMS); clf.fit(X_train_s, y_train)
    joblib.dump(clf, 'svm_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved.")