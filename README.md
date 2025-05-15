# Loan Approval Prediction using MLP

This project uses a **Multilayer Perceptron (MLP)** built with TensorFlow/Keras to classify loan applications as approved or not based on personal and loan-related features.

## 📁 Dataset

The model expects a CSV file named `loan_data.csv` with features such as:

* **Numerical**: `person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, `cb_person_cred_hist_length`, `credit_score`
* **Categorical**: `person_gender`, `person_education`, `person_home_ownership`, `loan_intent`, `previous_loan_defaults_on_file`
* **Target**: `loan_status` (binary classification: 0 = Rejected, 1 = Approved)

> **Note**: Missing values are dropped in this implementation.

---

## 🧠 Model Architecture

* Input Layer (normalized features)
* Dense Layer (64 neurons, ReLU) + Dropout (0.3)
* Dense Layer (32 neurons, ReLU) + Dropout (0.3)
* Output Layer (1 neuron, Sigmoid activation)

Optimizer: `Adam`
Loss Function: `Binary Crossentropy`
Metrics: `Accuracy`

---

## 🔧 Setup

Make sure you have the required libraries installed:

```bash
pip install pandas sklearn tensorflow matplotlib
```

---

## 🚀 How to Run

1. Upload your dataset to the same directory as the script or Colab notebook.
2. Ensure the filename is `loan_data.csv`.
3. Run the notebook or script.

---

## 📊 Outputs

* **Training and Validation Accuracy Plot**
* **Confusion Matrix**
* **Classification Report** (Precision, Recall, F1-score)
* **ROC Curve** with AUC score

---

## 🧪 Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* **ROC AUC Score**
* **Precision / Recall / F1-Score**

---

## 💾 Model Saving

The trained model is saved in two formats:

* `mlp.h5`
* `mlp.keras`

These can be reloaded later using:

```python
from tensorflow.keras.models import load_model
model = load_model('mlp.h5')  # or 'mlp.keras'
```

---

## 📌 Notes

* Categorical features are one-hot encoded using `pd.get_dummies`.
* Numerical features are standardized using `StandardScaler`.
* Uses `EarlyStopping` to avoid overfitting based on validation loss.

---

