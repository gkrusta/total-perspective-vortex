from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import numpy as np
import argparse
import matplotlib.pyplot as plt


def plot_label_distribution(features, labels):
    """
    Creates a 2D scatter plot of the features reduced to 2D using PCA,
    colored by the 3 labels to visualize class separation.
    """
    # Reduce dimensionality to 2D using PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(8, 6))
    colors = ['none', 'red', 'blue', 'green']  # Assuming labels are 0, 1, 2
    for label in np.unique(labels):
        mask = labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                    c=colors[label], label=f'Label {label}', alpha=0.7)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D PCA Projection of Features by Label')
    plt.legend()
    plt.grid(True)
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
    clf = LDA()
    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    y_pred = clf.predict(X_test)
    print("Classification report:", classification_report(y_test, y_pred))

    # Plot the label distribution
    plot_label_distribution(features, labels)


if __name__ == "__main__":
    main()
