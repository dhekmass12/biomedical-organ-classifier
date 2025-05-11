import os
import pandas as pd

def preprocess_train(train_csv, image_dir, output_pickle):
    """
    Identify missing views in train folder and save cleaned filenames/labels.
    """
    df = pd.read_csv(train_csv)
    missing = []
    valid = []
    for fname, cls in zip(df['id'], df['class']):
        base = fname.split('_', 1)[1]
        views = ['a_', 'c_', 's_']
        exists = {prefix: os.path.exists(os.path.join(image_dir, prefix + base)) for prefix in views}
        if not all(exists.values()):
            # log missing prefixes
            for prefix, ok in exists.items():
                if not ok:
                    missing.append((fname, prefix))
        else:
            valid.append((fname, cls))
    # Save missing and valid
    missing_df = pd.DataFrame(missing, columns=['csv_entry', 'missing_view'])
    missing_df.to_csv('missing_views.csv', index=False)
    pd.DataFrame(valid, columns=['id','class']).to_pickle(output_pickle)
    print(f"Missing view entries: {len(missing_df)} logged to missing_views.csv")
    print(f"Valid samples: {len(valid)} saved to {output_pickle}")

if __name__ == '__main__':
    preprocess_train(
        train_csv='organ-all-mnist-DL20242025/train.csv',
        image_dir='organ-all-mnist-DL20242025/train',
        output_pickle='/without_cnn/train_clean.pkl'
    )