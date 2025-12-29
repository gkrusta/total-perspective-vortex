from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import numpy as np
import argparse
import matplotlib.pyplot as plt
# from mne.decoding import CSP


def plot_label_distribution(features, labels):
    """
    Creates a 2D scatter plot of the features reduced to 2D using PCA,
    colored by the 3 labels to visualize class separation.
    """
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'blue', 'green']  # For labels 1, 2, 3
    label_names = {1: 'Label 1 (Rest)', 2: 'Label 2 (Left)', 3: 'Label 3 (Right)'}
    
    for label in np.unique(labels):
        if label == 0:
            continue  # Skip label 0 if it exists
        mask = labels == label
        ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=colors[label-1], label=label_names.get(label, f'Label {label}'), 
                   alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax.set_title('2D PCA Projection of Features by Class', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/pca_projection_2d.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: images/pca_projection_2d.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on EEG bandpower features and predict.")
    parser.add_argument("subject", type=str, help="Path to the subject's EEG data file.")
    parser.add_argument("run", type=str, help="One of the 14 runs.")
    parser.add_argument("task", type=str, choices=["train", "predict"], nargs='?', help="Train or predict.")
    
    features = np.load("data/X_train.npy")
    labels = np.load("data/y_train.npy")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=10)),
        ('lda', LDA())
    ])

    pipeline.fit(X_train, y_train)

    train_score = pipeline.score(X_train, y_train)
    print("Train score:", train_score)

    y_pred = pipeline.predict(X_test)
    print("Classification report:", classification_report(y_test, y_pred))

    # Plot the label distribution
    plot_label_distribution(features, labels)


if __name__ == "__main__":
    main()
