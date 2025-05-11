import os
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ================= User Config =================
IMAGE_DIR    = "organ-all-mnist-DL20242025/"      # contains 'train/' and 'test/' subfolders
TRAIN_CSV    = "organ-all-mnist-DL20242025/train.csv"   # has columns ['id','class']
TEST_CSV     = "organ-all-mnist-DL20242025/test.csv"    # has column ['id']
IMG_SIZE     = (128, 128)             # (height, width)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm='L2-Hys',
    feature_vector=True
)

SVM_PARAMS = dict(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,
    random_state=42
)

TEST_SUBMISSION_CSV = "submission_svm.csv"

def extract_multiview_hog_from_folder(filenames, folder, desc="Extracting HOG"):
    """
    Given a list of filenames (each starting with 'a_','c_' or 's_'),
    loads each view from IMAGE_DIR/folder/ as grayscale via cv2,
    resizes, extracts HOG, and returns an (N x D) array.
    Pads missing views with zeros of the correct HOG length.
    """
    base_dir = os.path.join(IMAGE_DIR, folder)

    # Precompute dummy HOG for padding:
    dummy_gray = np.zeros(IMG_SIZE, dtype=np.float32)
    dummy_hog  = hog(dummy_gray, **HOG_PARAMS)

    feats = []
    for full_fname in tqdm(filenames, desc=desc):
        # strip leading prefix to get the base name, e.g. "s_7351.png" -> "7351.png"
        _, base = full_fname.split("_", 1) if "_" in full_fname else ("", full_fname)

        view_feats = []
        for prefix in ("a_", "c_", "s_"):
            view_name = prefix + base
            view_path = os.path.join(base_dir, view_name)

            if not os.path.exists(view_path):
                # pad with dummy HOG for missing views
                view_feats.append(dummy_hog)
            else:
                gray = cv2.imread(view_path, cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    # treat unreadable as missing
                    view_feats.append(dummy_hog)
                else:
                    # normalize and resize
                    gray = gray.astype(np.float32) / 255.0
                    gray = cv2.resize(gray, (IMG_SIZE[1], IMG_SIZE[0]), interpolation=cv2.INTER_AREA)
                    view_feats.append(hog(gray, **HOG_PARAMS))

        feats.append(np.hstack(view_feats))

    return np.vstack(feats)


if __name__ == "__main__":
    # Load train and split into train/validation
    df = pd.read_csv(TRAIN_CSV)
    filenames = df['id'].values
    labels    = df['class'].values

    X_train_files, X_val_files, y_train, y_val = train_test_split(
        filenames, labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Extract HOG features with padding for missing views
    X_train = extract_multiview_hog_from_folder(
        X_train_files, folder="train",
        desc="HOG on train split"
    )
    X_val   = extract_multiview_hog_from_folder(
        X_val_files, folder="train",
        desc="HOG on val split"
    )

    print("\nScaling features…")
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # Train SVM
    print("Training SVM…")
    clf = SVC(**SVM_PARAMS)
    clf.fit(X_train_s, y_train)

    print("\nEvaluating on validation set…")
    y_pred = clf.predict(X_val_s)

    print(f"Validation Accuracy : {accuracy_score(y_val, y_pred):.4f}")
    print(f"Validation Precision: {precision_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"Validation Recall   : {recall_score(y_val, y_pred,    average='weighted'):.4f}")
    print(f"Validation F1-Score : {f1_score(y_val, y_pred,        average='weighted'):.4f}\n")

    print("Per-Class Report:\n", classification_report(y_val, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))

    # Predict test
    df_test    = pd.read_csv(TEST_CSV)
    test_files = df_test['id'].values

    X_test   = extract_multiview_hog_from_folder(
        test_files, folder="test",
        desc="HOG on test set"
    )
    X_test_s = scaler.transform(X_test)

    print("\nPredicting on test set…")
    test_preds = clf.predict(X_test_s)

    submission = pd.DataFrame({'id': test_files, 'class': test_preds})
    submission.to_csv(TEST_SUBMISSION_CSV, index=False)
    print(f"Submission saved to {TEST_SUBMISSION_CSV}")
