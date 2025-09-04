import numpy as np
import pandas as pd

class CurriculumHelper:
    def __init__(self, sensitive_feature):
        # Initialize with the sensitive feature
        self.sensitive_feature = sensitive_feature



    @staticmethod
    def  split_by_sensitive_feature_and_confidence(X, y_prob, sensitive_feature, n_splits=5, ascending_confindance=False):
        """
        Splits X and y into groups based on the given sensitive feature, orders them by their average prediction confidence,
        and sorts each group by confidence.

        ascending_confindance=False ⇒ groups with higher average confidence first ⇒ the easier (more confident) groups are provided to the student first.

        ascending_confindance=True ⇒ groups with lower average confidence first ⇒ the harder groups first.

        Returns:
            sensitive_feature_splits (list): A list of tuples where each tuple contains (X, y_pred) for each sensitive feature value.
        """
        # Get the maximum probability and corresponding predicted class
        max_prob = np.max(y_prob, axis=1)  # Get highest probability for each sample
        y_pred = np.argmax(y_prob, axis=1)  # Get predicted class labels

        # Add prediction data to X DataFrame
        X_sorted = X.copy()
        X_sorted['y_pred'] = y_pred
        X_sorted['max_prob'] = max_prob

        # Group by the sensitive feature (e.g., 'race', 'age_class', etc.)
        feature_groups = X_sorted.groupby(sensitive_feature)

        # Compute average confidence for each sensitive feature group
        feature_confidence = feature_groups["max_prob"].mean().sort_values(ascending=ascending_confindance)  

        # Store sensitive feature-based splits
        sensitive_feature_splits = []
        for feature_value in feature_confidence.index:
            feature_data = feature_groups.get_group(feature_value)

            # Drop auxiliary columns before returning
            split_X = feature_data.drop(columns=['y_pred', 'max_prob'])
            split_y_pred = feature_data['y_pred']

            # Append to the result list as a tuple
            sensitive_feature_splits.append((split_X, split_y_pred))

        return sensitive_feature_splits
        
    @staticmethod
    def split_into_difficulty_parts(X, y_prob, n_splits=5, ascending_confidance=False):
        """Splits X and y into 5 parts based on the highest prediction probability, 
        with the most difficult samples (lowest probability) getting split last.

        ascending_confindance=False ⇒ groups with higher average confidence first ⇒ the easier (more confident) groups are provided to the student first.

        ascending_confindance=True ⇒ groups with lower average confidence first ⇒ the harder groups first.

        """
        split_size = len(X) // n_splits
        
        # Get the maximum probability and corresponding predicted class
        max_prob = np.max(y_prob, axis=1)  # Get highest probability for each sample
        y_pred = np.argmax(y_prob, axis=1)  # Get predicted class labels

        # Combine X (DataFrame), y_pred, and max_prob into a single DataFrame for sorting
        X_sorted = X.copy()  # Work with a copy to avoid modifying original
        X_sorted['y_pred'] = y_pred
        X_sorted['max_prob'] = max_prob
        
        # By default: Sort by the highest probability (descending order) - more confident predictions come first
        X_sorted = X_sorted.sort_values(by='max_prob', ascending=ascending_confidance)

        #print(X_sorted.head())    
        # Split the sorted data into n_splits parts
        split_data = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X_sorted)
            split_X = X_sorted.iloc[start_idx:end_idx].drop(columns=['y_pred', 'max_prob'])  # Drop auxiliary columns
            #print(split_X.head())
            split_y_pred = X_sorted.iloc[start_idx:end_idx]['y_pred']
            split_data.append((split_X, split_y_pred))

        return split_data
    
    @staticmethod
    def split_into_difficulty_parts_asc(X, y_prob, n_splits=5):
        return CurriculumHelper.split_into_difficulty_parts(X, y_prob, n_splits, ascending_confidance=True)
    
    @staticmethod
    def split_into_parts(X, y_prob, n_splits=5):
        """Splits X and y into 5 equal parts and returns them as a list of (X_part, y_part)."""
        split_size = len(X) // n_splits
        
        # Convert probabilities to predictions
        y_pred = np.argmax(y_prob, axis=1)

        split_data = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(X)
            split_data.append((X[start_idx:end_idx], y_pred[start_idx:end_idx]))

        return split_data