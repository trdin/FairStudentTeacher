from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import signature


class TeacherStudentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, teacher, student, split_data=False):
        """
        Initialize the classifier with separate teacher and student models.

        Args:
        teacher_model: Classifier class for the teacher (e.g., RandomForestClassifier).
        student_model: Classifier class for the student (e.g., KNeighborsClassifier).
        """
        # Check if teacher model supports sample_weight
        fit_sig_teacher = signature(teacher.fit)
        if "sample_weight" not in fit_sig_teacher.parameters:
            raise ValueError(
                f"The teacher model '{teacher.__name__}' does not support sample weights."
            )
        
        # Check if student model supports sample_weight
        fit_sig_student = signature(student.fit)
        if "sample_weight" not in fit_sig_student.parameters:
            raise ValueError(
                f"The student model '{student.__name__}' does not support sample weights."
            )

        self.teacher = teacher
        self.student = student
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

        self.teacher.fit(X_teacher, y_teacher)

        # Predict using teacher model
        y_pred = self.teacher.predict(X_student)

        # Train student model
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
