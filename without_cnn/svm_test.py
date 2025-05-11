import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from skimage.feature import hog
import joblib

# Config
IMG_SIZE=(128,128)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys', feature_vector=True)

def extract_features(filenames, folder, image_base):
    dummy=np.zeros(IMG_SIZE, dtype=np.float32)
    dummy_hog=hog(dummy, **HOG_PARAMS)
    feats=[]
    for fname in tqdm(filenames, desc=f'HOG ({folder})'):
        base=fname.split('_',1)[1]
        view_feats=[]
        for p in ['a_','c_','s_']:
            path=os.path.join(image_base, folder, p+base)
            if not os.path.exists(path): view_feats.append(dummy_hog)
            else:
                g=cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
                g=cv2.resize(g,(IMG_SIZE[1],IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
                view_feats.append(hog(g, **HOG_PARAMS))
        feats.append(np.hstack(view_feats))
    return np.vstack(feats)

if __name__=='__main__':
    clf=joblib.load('svm_model.joblib')
    scaler=joblib.load('scaler.joblib')
    df_test=pd.read_csv('path/to/test.csv')
    files=df_test['id'].values
    X_test=extract_features(files, 'test', 'organ-all-mnist-DL20242025')
    X_test_s=scaler.transform(X_test)
    preds=clf.predict(X_test_s)
    pd.DataFrame({'id': files, 'class': preds}).to_csv('submission.csv', index=False)
    print("Submission.csv saved.")