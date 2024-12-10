# Fairness in Machine Learning: FairTeacherStudentClassifier

This project demonstrates a machine learning pipeline for addressing fairness concerns in classification tasks. Specifically, it implements a custom **FairTeacherStudentClassifier**, designed to identify and mitigate performance disparities across sensitive groups (e.g., race). The classifier trains a "student model" to account for the shortcomings of a "teacher model" on underrepresented or disadvantaged groups.

## Project Overview

The dataset used in this project is the **Adult Census Income dataset**, provided by the `fairlearn` library. The goal is to predict whether an individual earns more than $50K per year (`class`), using demographic and economic features (e.g., age, race, education).

### Key Features
- A **FairTeacherStudentClassifier** that:
  - Trains a teacher model to predict outcomes.
  - Identifies performance weaknesses of the teacher model across sensitive groups.
  - Adjusts the student model training by assigning weights to groups where the teacher performs poorly.
- Comparison of the FairTeacherStudentClassifier against a baseline model (RandomForestClassifier).
- Evaluation of fairness through accuracy metrics for each racial group.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Required libraries:
  - `fairlearn`
  - `scikit-learn`
  - `pandas`
  - `numpy`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fairness-ml.git
   cd fairness-ml
   ```
2. Install dependencies:
   ```bash
   pip install fairlearn scikit-learn pandas numpy
   ```

---

## Usage

### Step 1: Run the Code
Run the Python script to preprocess the data, train models, and evaluate performance:
```bash
python fairness_pipeline.py
```

### Step 2: Understand the Results
The script outputs a DataFrame comparing the accuracy of:
- The **FairTeacherStudentClassifier**.
- A **Baseline RandomForestClassifier**.

The comparison is broken down by sensitive groups (e.g., race).

---

## How It Works

### Dataset
The Adult Census Income dataset contains 48,842 samples with features such as:
- `age`, `workclass`, `education`, `occupation`, etc.
- Sensitive attribute: **race**.

### FairTeacherStudentClassifier Workflow
1. **Teacher Model**:
   - A base model is trained on the data.
   - Predictions are made on the training set.
2. **Performance Analysis**:
   - Sensitive groups are evaluated to identify where the teacher performs poorly.
   - Weights are assigned to these groups inversely proportional to their performance.
3. **Student Model**:
   - A student model is trained on the teacher’s predictions, using the group-specific weights.


---

## Customization

### Changing the Classifier
You can replace the default `RandomForestClassifier` with any other scikit-learn classifier (e.g., `SVC`, `GradientBoostingClassifier`).

### Changing the Sensitive Attribute
The project is designed to evaluate fairness based on the `race` attribute. However, you can easily change this to any other feature (e.g., `sex` or `education`) by modifying the `z` variable.

---

## Results and Observations

- The FairTeacherStudentClassifier consistently improves accuracy for underrepresented or disadvantaged groups compared to the baseline.
- It highlights the importance of fairness-aware training to mitigate bias in machine learning models.

---

## Contributing

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push the branch:
   ```bash
   git push origin feature-branch
   ```
5. Create a Pull Request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- [Fairlearn Documentation](https://fairlearn.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

Feel free to reach out for further clarifications or to report issues!