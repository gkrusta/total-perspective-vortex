from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
# from sklearn.decomposition import PCA
from pca import PCA
import numpy as np
import argparse
import matplotlib.pyplot as plt
from preprocess import DataLoader
# from mne.decoding import CSP


class ClassifierBCI:
    def __init__(self, subject=None, run=None, task=None):
        if subject and run and task:
            self.subject = subject.zfill(3)
            self.run = run.zfill(2)
            self.data = DataLoader("/sgoinfre/students/gkrusta/physionet.org/files/eegmmidb/1.0.0/S{self.subject}/S{self.subject}R{self.run}.edf")
            features = np.load("data/X_train.npy")
            labels = np.load("data/y_train.npy")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
            if task == "train":
                self.train(self.X_train, self.y_train)
            elif task == "predict":
                self.predict(self.X_test)
        else:
            experi


    def train(self, X_train, y_train):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10)),
            ('lda', LDA())
        ])
        self.pipeline.fit(X_train, y_train)
        train_score= self.pipeline.score(X_train, y_train)
        print("Train score:", train_score)
        # Flatten epochs/features to 2D for sklearn (n_samples, n_features)
        n_samples = features.shape[0]
        X = features.reshape(n_samples, -1)
        cv = min(5, n_samples)
        if cv < 2:
            cv = 2
        print(f"Running cross_val_score with cv={cv} on flattened features (shape {X.shape})")
        simple_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=10)),
            ('lda', LDA())
        ])
        scores = cross_val_score(simple_pipeline, X, labels, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f} (scores: {scores})")


    def predict(self, X):
        return self.pipeline.predict(X)


    def score(self, X, y):
        return self.pipeline.score(X, y)
    
    def experiment(self)
        ...
    

    


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on EEG bandpower features and predict.")
    parser.add_argument("subject", type=str, help="Path to the subject's EEG data file.")
    parser.add_argument("run", type=str, help="One of the 14 runs.")
    parser.add_argument("task", type=str, choices=["train", "predict"], nargs='?', help="Train or predict.")
    args = parser.parse_args()

    features = np.load("data/X_train.npy")
    labels = np.load("data/y_train.npy")
    
    if args.task and args.subject and args.run:
        print(f"Subject: {args.subject}, Run: {args.run}, Task: {args.task}")
        mybci = ClassifierBCI(subject=args.subject, run=args.run, task=args.task)
    if args == None:
        print("Running experiment with default settings.")
        mybci = ClassifierBCI()
    else:
        print("Missing arguments. Please provide subject, run, and task or no arguments at all.")
        return


    # proceed with a train/test split on the flattened data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    if args.task == "predict":
        print("Prediction mode not implemented yet.")
        return
    else:
        print("Training mode.")


    y_pred = pipeline.predict(X_test)
    print("Classification report:", classification_report(y_test, y_pred))

    # Plot the label distribution
    # plot_label_distribution(features, labels)
    print("variance ratio: ", np.sum(pipeline.named_steps['pca'].explained_variance_ratio))



if __name__ == "__main__":
    main()
