import os
import pandas as pd

def preprocess_train(train_csv, image_dir, output_pickle):
    """
    Identify missing views in train folder based on 'id' and save cleaned ids/labels.
    """
    df = pd.read_csv(train_csv)  # columns: ['id','class']
    missing = []
    valid = []
    for img_id, cls in zip(df['id'], df['class']):
        base = img_id.split('_', 1)[1]  # e.g., 'a_1.png' -> '1.png'
        views = ['a_', 'c_', 's_']
        exists = {prefix: os.path.exists(os.path.join(image_dir, prefix + base)) for prefix in views}
        if not all(exists.values()):
            for prefix, ok in exists.items():
                if not ok:
                    missing.append((img_id, prefix))
        else:
            valid.append((img_id, cls))
    missing_df = pd.DataFrame(missing, columns=['id', 'missing_view'])
    missing_df.to_csv('missing_views.csv', index=False)
    pd.DataFrame(valid, columns=['id','class']).to_pickle(output_pickle)
    print(f"Missing view entries: {len(missing_df)} logged to missing_views.csv")
    print(f"Valid samples: {len(valid)} saved to {output_pickle}")

if __name__ == '__main__':
    preprocess_train(
        train_csv='organ-all-mnist-DL20242025/train.csv',
        image_dir='organ-all-mnist-DL20242025/train',
        output_pickle='train_clean.pkl'
    )