from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from inspect import signature


class CurriculumStudentTeacher(BaseEstimator, ClassifierMixin):
    def __init__(self, teacher_type, student_type, transform_func, split_data=False, n_splits=5, random_state=40, shuffle=True):
        """
        Initialize the classifier with specific teacher and student types.

        Args:
        teacher_type: Classifier class for the teacher model.
        student_type: Classifier class for the student model (must support partial_fit).
        transform_func: Function that takes X and y (teacher predictions) and returns multiple X, y sets.
        split_data: Whether to split the data into teacher and student sets.
        """
        self.teacher_type = teacher_type
        self.student_type = student_type
        self.transform_func = transform_func
        self.split_data = split_data
        self.n_splits = n_splits
        self.random_state = random_state
        self.shuffle = shuffle
        self.teacher = None
        self.student = None

        # Check if the student supports partial_fit
        if not hasattr(student_type(), "partial_fit"):
            raise ValueError(
                f"The student model '{student_type.__name__}' must support partial_fit."
            )

    def fit(self, X, y):
        """
        Fit the teacher model, identify weaknesses, and train the student model incrementally.

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

        # Train the teacher model
        self.teacher = self.teacher_type()
        self.teacher.fit(X_teacher, y_teacher)

        # Predict using the teacher model
        y_prob = self.teacher.predict_proba(X_student)

        # Transform the teacher's predictions
        transformed_data = self.transform_func(X_student, y_prob, self.n_splits)

        # Train the student model incrementallys
        if not hasattr(self.student_type(), "random_state"):
            self.student = self.student_type()
        else:
            #TODO add LogLoss - logisticna regressija- bi naj bla bolj primerna  
            self.student = self.student_type(random_state=self.random_state, shuffle=self.shuffle, average=True)



        classes = np.unique(y)  # Ensure partial_fit is aware of all classes
        for X_part, y_part in transformed_data:
            self.student.partial_fit(X_part, y_part, classes=classes)

    def predict(self, X):
        """
        Predict using the student model.

        Args:
        X: Feature matrix (2D array-like).

        Returns:
        Predictions from the student model.
        """
        return self.student.predict(X)
