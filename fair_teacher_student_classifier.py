from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

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

    def fit(self, X, y, z):
        """
        Fit the teacher model, identify weaknesses, and train the student model.

        Args:
        X: Feature matrix (2D array-like).
        y: Target labels (1D array-like).
        z: Sensitive group labels (1D array-like, e.g., race).
        """
        # Convert z to a numpy array
        z = np.array(z)

        # Initialize and train the teacher model
        self.teacher = self.classifier_type()
        self.teacher.fit(X, y)

        # Predict using the teacher model
        y_pred = self.teacher.predict(X)

        unique_groups = np.unique(z)
        group_weights = {}

        for group in unique_groups:
            group_mask = (z == group)
            group_accuracy = accuracy_score(y[group_mask], y_pred[group_mask])

            # Lower accuracy -> higher weight
            group_weights[group] = 1 - group_accuracy + 1

        sample_weights = np.array([group_weights[group] for group in z])

        self.student = self.classifier_type()
        self.student.fit(X, y_pred, sample_weight=sample_weights)

    def predict(self, X):
        """
        Predict using the student model.

        Args:
        X: Feature matrix (2D array-like).

        Returns:
        Predictions from the student model.
        """
        return self.student.predict(X)
