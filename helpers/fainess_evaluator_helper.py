from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

class FairnessEvaluatorHelper:
    def __init__(self, folds, predictions):
        """
        Initialize the FairnessEvaluator with folds and model predictions.

        :param folds: List of (X_train, X_test, y_train, y_test, z_train, z_test) tuples from k-fold cross-validation.
        :param predictions: Dictionary of model names mapping to lists of predictions (one per fold).
        """
        self.folds = folds
        self.predictions = predictions
        self.models = list(predictions.keys())

        # Store fairness metrics for each model
        self.accuracy_differences = {model: [] for model in self.models}
        self.demographic_parity_differences = {model: [] for model in self.models}
        self.equalized_odds_differences = {model: [] for model in self.models}
        self.accuracies = {model: [] for model in self.models}  # New: Store accuracies per model
        self.group_accuracies = {}  # Accuracy per sensitive feature group

    def compute_fairness_metrics(self):
        """
        Calculate accuracy differences, demographic parity differences, 
        and equalized odds differences across all folds.
        """
        for fold_idx, (X_train, X_test, y_train, y_test, z_train, z_test) in enumerate(self.folds):
            for model_name, prediction_list in self.predictions.items():
                # Get the predictions for the current fold
                fold_pred = prediction_list[fold_idx]

                # Compute accuracy and fairness metrics
                metric_frame = MetricFrame(
                    metrics=accuracy_score,
                    y_true=y_test,
                    y_pred=fold_pred,
                    sensitive_features=z_test
                )
                self.accuracy_differences[model_name].append(metric_frame.difference())
                self.demographic_parity_differences[model_name].append(
                    demographic_parity_difference(y_test, fold_pred, sensitive_features=z_test)
                )
                self.equalized_odds_differences[model_name].append(
                    equalized_odds_difference(y_test, fold_pred, sensitive_features=z_test)
                )
                self.accuracies[model_name].append(accuracy_score(y_test, fold_pred))  # Store accuracy

                # Compute accuracy per group
                for group in np.unique(z_test):
                    mask = (z_test == group)
                    group_accuracy = accuracy_score(y_test[mask], fold_pred[mask])

                    if group not in self.group_accuracies:
                        self.group_accuracies[group] = {model_name: []}
                    elif model_name not in self.group_accuracies[group]:
                        self.group_accuracies[group][model_name] = []

                    self.group_accuracies[group][model_name].append(group_accuracy)

    def sort_metric(self, metric_dict):
        """
        Sorts models based on the mean value of the fairness metric.

        :param metric_dict: Dictionary containing metric values per model
        :return: Sorted list of model names and corresponding metric values
        """
        metric_means = {model: np.mean(values) for model, values in metric_dict.items()}
        sorted_models = sorted(metric_means.items(), key=lambda x: x[1])  # Sort by fairness (lower is better)
        sorted_names = [item[0] for item in sorted_models]
        sorted_values = [metric_dict[item[0]] for item in sorted_models]
        return sorted_names, sorted_values

    def plot_metric(self, metric_dict, title, xlabel, color, sort_ascending=True):
        """
        Generalized function to plot a sorted box plot for a given metric.

        :param metric_dict: Dictionary containing metric values per model
        :param title: Title of the plot
        :param xlabel: Label for the x-axis
        :param color: Color of the box plot
        :param sort_ascending: Whether to sort in ascending order (lower is better)
        """
        if not metric_dict:
            raise ValueError(f"You must call compute_fairness_metrics() before plotting {title}.")

        # Sort models by mean fairness metric
        sorted_models, sorted_values = self.sort_metric(metric_dict)
        if not sort_ascending:
            sorted_models.reverse()
            sorted_values.reverse()

        # Create the box plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=sorted_values, vert=False, orient='h', color=color)

        # Set axis labels and title
        plt.xlabel(xlabel)
        plt.ylabel("Pristopi")
        plt.title(title)

        # Annotate with the sorted model names
        plt.yticks(ticks=range(len(sorted_models)), labels=sorted_models)

        plt.tight_layout()
        plt.show()

        # ---- MARKDOWN TABELA ----
        headers = [
            "Model",
            "Vrednosti (K-fold)",
            "Povprečje",
            "Mediana",
            "Minimum",
            "Maksimum",
        ]

        lines = []
        lines.append("|" + "|".join(headers) + "|")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for name, vals in zip(sorted_models, sorted_values):
            arr = np.array(vals)
            povp = np.mean(arr)
            med = np.median(arr)
            minv = np.min(arr)
            maxv = np.max(arr)

            # vrednosti ločene s ;
            vrednosti_str = "; ".join(f"{v:.4f}" for v in arr)

            row = [
                name,
                vrednosti_str,
                f"{povp:.4f}",
                f"{med:.4f}",
                f"{minv:.4f}",
                f"{maxv:.4f}",
            ]
            lines.append("|" + "|".join(row) + "|")

        md_table = "\n".join(lines)
        print(md_table)


    def plot_accuracy_differences(self):
        """ Plot a sorted box plot for accuracy differences. """
        self.plot_metric(
            self.accuracy_differences, 
            "Razlika v točnosti (Primerjava pravičnosti)", 
            "Razlika v točnosti", 
            "skyblue"  # Different color
        )

    def plot_demographic_parity(self):
        """ Plot a sorted box plot for demographic parity differences. """
        self.plot_metric(
            self.demographic_parity_differences, 
            "Razlika Demografkse  paritete (Primerjava pravičnosti)", 
            "Demografska pariteta", 
            "lightcoral"  # Different color
        )

    def plot_equalized_odds(self):
        """ Plot a sorted box plot for equalized odds differences. """
        self.plot_metric(
            self.equalized_odds_differences, 
            "Razlika Metrike uravnoteženih verjetnosti (Primerjava pravičnosti)", 
            "Razlika Metrike uravnoteženih verjetnosti", 
            "lightgreen"  # Different color
        )

    def plot_accuracies(self):
        """ Plot a sorted box plot for accuracy distributions. """
        self.plot_metric(
            self.accuracies, 
            "Primerjava Točnosti", 
            "Točnost", 
            "gold",  # Unique color for accuracy
            sort_ascending=False    # Higher accuracy is better
        )

    
    def plot_group_accuracies(self, label_encoder):
        """
        Plot accuracy distribution (box plots) for each sensitive feature group.

        :param label_encoder: LabelEncoder used for encoding the sensitive feature (e.g., race).
        """
        if not self.group_accuracies:
            raise ValueError("You must call compute_fairness_metrics() before plotting group accuracies.")

        for group, model_accuracies in self.group_accuracies.items():
            # Convert dictionary to DataFrame for Seaborn
            df = pd.DataFrame.from_dict(model_accuracies, orient='index').T

            # Sort models by median accuracy before plotting
            sorted_models = df.median().sort_values(ascending=False).index
            df = df[sorted_models]

            # Decode the sensitive feature group name using label_encoder
            group_name = label_encoder.inverse_transform([group])[0]

            # Create the box plot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, orient="h", palette="Set2")

            # Set labels and title
            plt.xlabel("Točnost")
            plt.ylabel("Pristopi")
            plt.title(f"Porazdelitev točnosti po modelu za občutljiv spremenljivko {group_name}")

            # Get the min and max accuracy values
            min_accuracy = df.min().min()
            max_accuracy = df.max().max()

            # Add 0.05 to the min and max accuracy for setting chart limits
            min_limit = max(0, min_accuracy - 0.05)  # Ensure the min limit doesn't go below 0
            max_limit = min(1, max_accuracy + 0.05)  # Ensure the max limit doesn't exceed 

            # Set consistent x-axis limits (adjust as needed)
            plt.xlim(min_limit, max_limit)

            plt.tight_layout()
            plt.show()

    def plot_selected_model_accuracies_by_sensetive_feature(self, selected_models, label_encoder):
        """
        Plot the accuracies for selected models across different sensitive feature groups (e.g., race).
        
        :param selected_models: List of model names to include in the plots.
        :param label_encoder: LabelEncoder for decoding sensitive feature names (e.g., race).
        """
        if not self.group_accuracies:
            raise ValueError("You must call compute_fairness_metrics() before plotting group accuracies.")

        # Filter the accuracy dictionary to include only the selected models
        filtered_accuracies = {model: self.accuracy_differences[model] for model in selected_models}
        
        # Sort the selected models by the mean accuracy (descending order)
        sorted_accuracies = sorted(filtered_accuracies.items(), key=lambda x: sum(x[1])/len(x[1]), reverse=True)
        
        # Extract the sorted model names
        sorted_models = [item[0] for item in sorted_accuracies]

        results_df = pd.DataFrame.from_dict(self.group_accuracies, orient='index')
        
        # Plot accuracies by race for each selected model
        for model in sorted_models:
            model_accuracies = results_df[model]

            # Decode the sensitive feature names (e.g., race) for display
            race_names = label_encoder.inverse_transform(model_accuracies.index)

            # Convert into a DataFrame
            df = pd.DataFrame(model_accuracies.tolist(), index=race_names)
            df = df.reset_index().melt(id_vars=["index"], var_name="Trial", value_name="Accuracy")
            df.rename(columns={"index": "Sensetive Feature"}, inplace=True)

            # Sort races by mean accuracy (descending order)
            race_order = df.groupby("Sensetive Feature")["Accuracy"].mean().sort_values(ascending=False).index
            df["Sensetive Feature"] = pd.Categorical(df["Sensetive Feature"], categories=race_order, ordered=True)

            # Create the boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df, x="Accuracy", y="Sensetive Feature", legend=False, hue="Sensetive Feature",  color="lightgreen",palette="Set2")

            # Set the labels and title
            plt.xlabel("Accuracy")
            plt.ylabel("Sensitive Feature Groups")
            plt.title(f"{model} Accuracy Across Sensitive Feature Groups")

            # Set the accuracy range
            plt.xlim(0.75, 1)  # Adjust accuracy range as needed

            # Display the plot
            plt.tight_layout()
            plt.show()

