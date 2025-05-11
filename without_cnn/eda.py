import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def eda_summary(pickle_file, missing_csv):
    # Load cleaned and missing data
    df_valid = pd.read_pickle(pickle_file)
    df_missing = pd.read_csv(missing_csv)

    # Overview
    total_raw = len(df_valid) + len(df_missing['csv_entry'].unique())
    total_valid = len(df_valid)
    total_missing = df_missing['csv_entry'].nunique()

    print(f"Total entries in raw train.csv: {total_raw}")
    print(f"Valid samples after preprocessing: {total_valid}")
    print(f"Entries with missing views: {total_missing}")

    # Show missing entries sample
    print("\nSample missing entries (first 10):")
    print(df_missing.head(10))

    # Class distribution before and after
    raw_counts = df_missing['csv_entry'].value_counts().rename('missing').to_frame()
    valid_counts = df_valid['class'].value_counts().rename('valid').to_frame()
    dist = valid_counts.join(raw_counts, how='outer').fillna(0)

    print("\nClass distribution (valid vs missing):")
    print(dist)

    # Plot valid class distribution
    plt.figure(figsize=(10,6))
    sns.countplot(x='class', data=df_valid)
    plt.title('Valid Samples Class Distribution')
    plt.savefig('class_distribution_valid.png')
    print("Saved class_distribution_valid.png")

if __name__ == '__main__':
    eda_summary(
        pickle_file='/without_cnn/train_clean.pkl',
        missing_csv='/without_cnn/missing_views.csv'
    )