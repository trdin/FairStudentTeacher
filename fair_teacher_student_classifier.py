from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split

class FairTeacherStudentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier_type):
        """
        Initialize the classifier with a specific classifier type.
        
        Args:
        classifier_type: Classifier class (e.g., RandomForestClassifier, KNeighborsClassifier).
        """
        self.classifier_type = classifier_type
        self.teacher = None
        self.student = None

    def fit(self, X, y, z, mode=0):
        """
        Fit the teacher model, identify weaknesses, and train the student model.

        Args:
        X: Feature matrix (2D array-like).
        y: Target labels (1D array-like).
        z: Sensitive group labels (1D array-like, e.g., race).
        """
        z = np.array(z)

        # initialize and train the teacher model
        self.teacher = self.classifier_type()
        self.teacher.fit(X, y)

        # predict using the teacher model
        y_pred = self.teacher.predict(X)

        unique_groups = np.unique(z)
        group_weights = {}
        group_accuracies = {}

        for group in unique_groups:
            group_mask = (z == group)
            group_accuracy = accuracy_score(y[group_mask], y_pred[group_mask])
            group_accuracies[group] = group_accuracy

        if mode == 0:
            # Lower accuracy -> higher weight for all groups
            group_weights = {group: 1 - acc + 1 for group, acc in group_accuracies.items()}
        elif mode == 1:
            # Set weight to 1.25 for the least accurate group
            min_accuracy_group = min(group_accuracies, key=group_accuracies.get)
            print(f"Min accuracy group: {min_accuracy_group}")
            group_weights = {group: (1 if group != min_accuracy_group else 1.25) for group in unique_groups}
        else:
            raise ValueError("Invalid mode. Choose 0 or 1.")

        sample_weights = np.array([group_weights[group] for group in z])

        self.student = self.classifier_type()
        self.student.fit(X, y_pred, sample_weight=sample_weights)

    
    def fit_split(self, X, y, z, mode=0):
        """
        Fit the teacher model, identify weaknesses, and train the student model.

        Args:
        X: Feature matrix (2D array-like).
        y: Target labels (1D array-like).
        z: Sensitive group labels (1D array-like, e.g., race).
        mode: Mode of weighting (0 or 1).
        """
        z = np.array(z)

        # Split data for teacher and student
        X_teacher, X_student, y_teacher, y_student, z_teacher, z_student = train_test_split(
            X, y, z, test_size=0.5, stratify=z, random_state=42
        )

        # Initialize and train the teacher model
        self.teacher = self.classifier_type()
        self.teacher.fit(X_teacher, y_teacher)

        # Predict using the teacher model on the student data
        y_pred = self.teacher.predict(X_student)

        unique_groups = np.unique(z_student)
        group_accuracies = {}

        for group in unique_groups:
            group_mask = (z_student == group)
            group_accuracy = accuracy_score(y_student[group_mask], y_pred[group_mask])
            group_accuracies[group] = group_accuracy

        if mode == 0:
            # Lower accuracy -> higher weight for all groups
            group_weights = {group: 1 - acc + 1 for group, acc in group_accuracies.items()}
        elif mode == 1:
            # Set weight to 1.25 for the least accurate group
            min_accuracy_group = min(group_accuracies, key=group_accuracies.get)
            print(f"Min accuracy group: {min_accuracy_group}")
            group_weights = {group: (1 if group != min_accuracy_group else 1.25) for group in unique_groups}
        else:
            raise ValueError("Invalid mode. Choose 0 or 1.")

        sample_weights = np.array([group_weights[group] for group in z_student])

        self.student = self.classifier_type()
        self.student.fit(X_student, y_pred, sample_weight=sample_weights)


    def predict(self, X):
        """
        Predict using the student model.

        Args:
        X: Feature matrix (2D array-like).

        Returns:
        Predictions from the student model.
        """
        return self.student.predict(X)
