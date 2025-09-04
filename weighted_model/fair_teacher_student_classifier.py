from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import signature


class FairTeacherStudentClassifier(BaseEstimator, ClassifierMixin):
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

    def fit(self, X, y, z, mode=0):
        z = np.array(z)

        if self.split_data:
            # Split data for teacher and student
            X_teacher, X_student, y_teacher, y_student, z_teacher, z_student = (
                train_test_split(X, y, z, test_size=0.5, stratify=z, random_state=42)
            )
        else:
            # Don't split the data
            X_teacher, y_teacher, z_teacher = X, y, z
            X_student, y_student, z_student = X, y, z

        # Initialize and train the teacher model
        self.teacher.fit(X_teacher, y_teacher)

        # Predict using the teacher model on the student data
        y_pred = self.teacher.predict(X_student)

        unique_groups = np.unique(z_student)
        group_accuracies = {}

        for group in unique_groups:
            group_mask = z_student == group
            group_accuracy = accuracy_score(y_student[group_mask], y_pred[group_mask])
            group_accuracies[group] = group_accuracy

        if mode == 0:
            # Lower accuracy -> higher weight for all groups
            group_weights = {
                group: 1 - acc for group, acc in group_accuracies.items()
            }
        elif mode == 1:
            # Set weight to 1.25 for the least accurate group
            min_accuracy_group = min(group_accuracies, key=group_accuracies.get)
            print(f"Min accuracy group: {min_accuracy_group}")
            group_weights = {
                group: (1 if group != min_accuracy_group else 1.25)
                for group in unique_groups
            }
        elif mode == 2:
            # Maximum accuracy group - accuracy for current group
            max_accuracy = max(group_accuracies.values())
            group_weights = {
                group: max_accuracy - acc for group, acc in group_accuracies.items()
            }
        else:
            raise ValueError("Invalid mode. Choose 0, 1 or 2.")
        
        # Normalize weights using numpy clip
        total_weight = sum(group_weights.values())
        for group in group_weights:
            group_weights[group] = np.clip(
                group_weights[group] / total_weight, 0.01, 1.0
            )  # Preventing zero weights

        sample_weights = np.array([group_weights[group] for group in z_student])

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
