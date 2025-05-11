import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_summary(pickle_file, missing_csv, raw_csv):
    df_raw = pd.read_csv(raw_csv)            # columns ['id','class']
    df_valid = pd.read_pickle(pickle_file)   # columns ['id','class']
    df_missing = pd.read_csv(missing_csv)    # columns ['id','missing_view']

    # Overall counts
    total_raw = len(df_raw)
    total_valid = len(df_valid)
    total_missing = df_missing['id'].nunique()

    print(f"Total entries in raw train.csv: {total_raw}")
    print(f"Valid samples after preprocessing: {total_valid}")
    print(f"Entries with missing views: {total_missing}")

    print("\nSample missing entries (first 10):")
    print(df_missing.head(10))

    # Class distribution before and after
    raw_dist = df_raw['class'].value_counts().rename('raw')
    valid_dist = df_valid['class'].value_counts().rename('valid')
    dist_df = pd.concat([raw_dist, valid_dist], axis=1).fillna(0).astype(int)

    print("\nClass distribution (raw vs valid):")
    print(dist_df)

    # Plot raw vs valid class distribution
    dist_df.plot(kind='bar', figsize=(10,6))
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Raw vs Valid Samples Class Distribution')
    plt.tight_layout()
    plt.savefig('class_distribution_comparison.png')
    print("Saved class_distribution_comparison.png")

    # Missing views per plane by class
    # Merge class into missing
    df_missing = df_missing.merge(df_raw[['id','class']], on='id', how='left')
    missing_summary = df_missing.groupby(['class','missing_view']).size().unstack(fill_value=0)

    # Plot stacked bar
    missing_summary.plot(kind='bar', stacked=True, figsize=(12,6))
    plt.xlabel('Class')
    plt.ylabel('Number of Missing Views')
    plt.title('Missing Views per Plane by Class')
    plt.legend(title='View', labels=['Axial (a)','Coronal (c)','Sagittal (s)'])
    plt.tight_layout()
    plt.savefig('missing_views_by_class.png')
    print("Saved missing_views_by_class.png")

if __name__ == '__main__':
    eda_summary(
        pickle_file='train_clean.pkl',
        missing_csv='missing_views.csv',
        raw_csv='organ-all-mnist-DL20242025/train.csv'
    )