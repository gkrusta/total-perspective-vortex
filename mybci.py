from os import pread
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, ShuffleSplit
# from sklearn.decomposition import PCA
from pca import PCA
import numpy as np
import argparse
import matplotlib.pyplot as plt
from preprocess import DataLoader
from utils import EXPERIMENTS, USEFULL_CHANNELS, find_events
# from mne.decoding import CSP


# channel selection lacking

class ClassifierBCI:
    def __init__(self, subject=None, run=None, task=None, model="LDA"):
        print("EXPERIMENTS:", EXPERIMENTS.keys())
        if subject and run and task:
            model = DataLoader(subject, run)
            # maybe remove np save from save features function
            # Filter to include only annotations from t1 and t2, excluding t0
            mask = (model.labels == 2) | (model.labels == 3)
            self.features = model.features[mask]
            self.labels = model.labels[mask]
            self.model = model
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.features,
                self.labels,
                test_size=0.2,
                random_state=42,
                stratify=self.labels
            )
            if task == "train":
                self.train(self.X_train, self.y_train)
            elif task == "predict":
                self.predict()
        else:
            self.experiment()


    def train(self, X_train, y_train):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=4)),
            ('lda', LDA())
        ])
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        features = np.log(X_train.mean(axis=2))
        print("X_train shape:", X_train.shape)
        print("features shape:", features.shape)
        self.pipeline.fit(features, y_train)
        train_score= self.pipeline.score(features, y_train)
        print("Train score:", train_score)

        print(f"Running cross_val_score with cv=10 on flattened features (shape {self.X_train.shape})")

        scores = cross_val_score(self.pipeline, features, self.y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f"Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f} (scores: {scores})")
        return scores.mean()


    def predict(self):
        y_pred = self.pipeline.predict(self.X_test)
        print("Classification report:", classification_report(self.y_test, y_pred))
        # Plot the label distribution
        # plot_label_distribution(features, labels)
        print("variance ratio: ", np.sum(self.pipeline.named_steps['pca'].explained_variance_ratio))
        print("Predictions:", y_pred)
        return y_pred


    def score(self, X, y):
        return self.pipeline.score(X, y)
    

    def experiment(self):
        """
        Run experiments over all subjects and runs defined in EXPERIMENTS.
        Computes and prints the accuracy for each experiment in each subject
        and across all subjects an mean accuracy in general.
        """
        accuracy_of_experiments = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for exp_name, exp_info in EXPERIMENTS.items():
            for subject in range(1, 109):
                accuracy_of_all_runs = []
                for run in exp_info['runs']:
                    try:

                        model = DataLoader(subject, run)
                        mask = find_events(model, exp_info)
                        features = model.features[mask]
                        labels = model.labels[mask]
                        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            features,
                            labels,
                            test_size=0.2,
                            random_state=42,
                            stratify=labels
                        )
                        acc_of_1_run = self.train(self.X_train, self.y_train)
                        # acc_of_1_run = self.score(self.X_test, self.y_test)
                        acc = self.predict()
                        features = np.log(self.X_train.mean(axis=2))
                        print("acc_of_1_run:")
                        accuracy_of_all_runs[exp_name].append(acc_of_1_run)
                    except Exception as e:
                        print(f"Error processing Subject {subject}, Run {run}, Experiment {exp_name}: {e}")
                        exit(1)
                print("accuracy of all runs:", accuracy_of_all_runs[exp_name])
                acc_of_1_subject = np.mean(accuracy_of_all_runs[exp_name])
                print(f"experiment {exp_name}: subject {subject}: accuracy: {acc_of_1_subject:.3f}")
                accuracy_of_experiments[exp_name].append(acc_of_1_subject)

            print("Mean accuracy of the six different experiments for all 109 subjects:")
            mean_accuracy = []
            for exp_name, _ in EXPERIMENTS.items():
                acc_of_1_experiment = np.mean(accuracy_of_experiments[exp_name])
                mean_accuracy.append(acc_of_1_experiment)
                print(f"experiment {exp_name}:     accuracy = {acc_of_1_experiment:.3f}")
            print(f"Mean accuracy of 6 experiments: {np.mean(mean_accuracy):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on EEG bandpower features and predict.")
    parser.add_argument("subject", type=str, nargs='?', help="Path to the subject's EEG data file.")
    parser.add_argument("run", type=str, nargs='?', help="One of the 14 runs.")
    parser.add_argument("task", type=str, nargs='?', choices=["train", "predict"], help="Train or predict.")
    parser.add_argument("model", type=str, nargs='?', help="Type of model to train. Default is LDA.", default="LDA")
    args = parser.parse_args()

    
    if args.task and args.subject and args.run:
        print(f"Subject: {args.subject}, Run: {args.run}, Task: {args.task}, Model: {args.model}")
        mybci = ClassifierBCI(subject=args.subject, run=args.run, task=args.task, model=args.model)
    elif not args.task and not args.subject and not args.run:
        print("Running experiment with default settings.")
        mybci = ClassifierBCI(model=args.model)
    else:
        print("Missing arguments. Please provide subject, run, and task or no arguments at all for experiment.")
        return


if __name__ == "__main__":
    main()
