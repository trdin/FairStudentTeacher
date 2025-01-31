from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import signature


class TeacherStudentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier_type, split_data=False):
        """
        Initialize the classifier with a specific classifier type.

        Args:
        classifier_type: Classifier class (e.g., RandomForestClassifier, KNeighborsClassifier).
        """

        # check if classifier_type supports sample_weight
        fit_sig = signature(classifier_type().fit)
        if "sample_weight" not in fit_sig.parameters:
            raise ValueError(
                f"The classifier_type '{classifier_type.__name__}' does not support sample weights."
            )

        self.classifier_type = classifier_type  # preveri će podpira weights čene error
        self.teacher = None
        self.student = None
        self.split_data = split_data

    def fit(self, X, y):
        """
        Fit the teacher model, identify weaknesses, and train the student model.

        Args:
        X: Feature matrix (2D array-like).
        y: Target labels (1D array-like).
        """
        if self.split_data:
            X_teacher, X_student, y_teacher, y_student = train_test_split(
                X, y, test_size=0.5, random_state=42
            )
        else:
            X_teacher, y_teacher = X, y
            X_student, y_student = X, y

        self.teacher = self.classifier_type()
        self.teacher.fit(X_teacher, y_teacher)

        # Predict using teacher model
        y_pred = self.teacher.predict(X_student)

        # Train student model
        self.student = self.classifier_type()
        self.student.fit(X_student, y_pred)

    def predict(self, X):
        """
        Predict using the student model.

        Args:
        X: Feature matrix (2D array-like).

        Returns:
        Predictions from the student model.
        """
        return self.student.predict(X)
