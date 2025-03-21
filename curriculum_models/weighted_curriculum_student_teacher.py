from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import signature
from fairlearn.metrics import MetricFrame


class WeightedCurriculumStudentTeacher(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        teacher,
        student,
        transform_func,
        split_data=False,
        n_splits=5,
        random_state=40,
        shuffle=True,
    ):
        """
        Initialize the classifier with specific teacher and student models.

        Args:
        teacher: Initialized classifier instance for the teacher model.
        student: Initialized classifier instance for the student model (must support partial_fit).
        transform_func: Function that takes X and y (teacher predictions) and returns multiple X, y sets.
        split_data: Whether to split the data into teacher and student sets.
        """
        self.teacher = teacher
        self.student = student
        self.transform_func = transform_func
        self.split_data = split_data
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle

        # Check if the student supports partial_fit
        if not hasattr(self.student, "partial_fit"):
            raise ValueError(
                f"The student model '{type(self.student).__name__}' must support partial_fit."
            )

    # Possible to check and adjust the wieghts every partial fit, but maybe this is not necessary since the model probably does this on its own??
    def fit(self, X, y, z, mode=0):
        z = np.array(z)

        if self.split_data:
            # Split data for teacher and student
            X_teacher, X_student, y_teacher, y_student, z_teacher, z_student = (
                train_test_split(X, y, z, test_size=0.5, stratify=z, random_state=42)
            )
        else:
            # dont split the data
            X_teacher, y_teacher, z_teacher = X, y, z
            X_student, y_student, z_student = X, y, z

        # Train the teacher model
        #self.teacher = self.teacher_type()
        self.teacher.fit(X_teacher, y_teacher)

        # Predict using the teacher model
        y_prob = self.teacher.predict_proba(X_student)
        sample_weights = self.get_sample_weights(mode, y_student, z_student, y_prob)

        X_student["weight"] = sample_weights

        # Transform the teacher's predictions
        transformed_data = self.transform_func(X_student, y_prob, self.n_splits)

        # Train the student model incrementallys
        """ if not hasattr(self.student_type(), "random_state"):
            self.student = self.student_type()
        else:
            self.student = self.student_type(
                random_state=self.random_state, shuffle=self.shuffle, average=True
            ) """

        classes = np.unique(y)  # Ensure partial_fit is aware of all classes
        for X_part, y_part in transformed_data:
            weights = X_part["weight"]
            X_part.drop(columns=["weight"], inplace=True)
            self.student.partial_fit(
                X_part, y_part, classes=classes, sample_weight=weights
            )

    def get_sample_weights(self, mode, y_student, z_student, y_prob):
        y_pred = np.argmax(y_prob, axis=1)

        unique_groups = np.unique(z_student)

        metric_frame = MetricFrame(
            metrics=accuracy_score,
            y_true=y_student,
            y_pred=y_pred,
            sensitive_features=z_student,
        )

        group_accuracies = metric_frame.by_group.to_dict()

        if mode == 0:
            # Lower accuracy -> higher weight for all groups
            group_weights = {group: 1 - acc for group, acc in group_accuracies.items()}
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
            raise ValueError("Invalid mode. Choose 0, 1, or 2.")

        # Normalize weights using numpy clip
        total_weight = sum(group_weights.values())
        for group in group_weights:
            group_weights[group] = np.clip(
                group_weights[group] / total_weight, 0.01, 1.0
            )  # Preventing zero weights

        sample_weights = np.array([group_weights[group] for group in z_student])
        return sample_weights

    def predict(self, X):
        """
        Predict using the student model.

        Args:
        X: Feature matrix (2D array-like).

        Returns:
        Predictions from the student model.
        """
        return self.student.predict(X)
