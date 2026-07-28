# 🚨 Cyberbullying-Detection-using-SVM

A multi-label Natural Language Processing (NLP) project that detects different types of toxic and cyberbullying comments using **TF-IDF** feature extraction and **Support Vector Machines (SVM)**.

## 📌 Project Overview

Online platforms receive millions of comments every day, making manual moderation difficult. This project automatically classifies comments into multiple toxicity categories using classical machine learning techniques.

The model predicts whether a comment belongs to one or more of the following categories:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

---

## 📂 Dataset

**Dataset:** Jigsaw Toxic Comment Classification Dataset

Each comment can belong to multiple categories simultaneously, making this a **multi-label text classification** problem.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* NLTK
* Scikit-learn
* TF-IDF Vectorizer
* Linear Support Vector Machine (LinearSVC)

---

## 📊 Exploratory Data Analysis

The notebook includes:

* Dataset auditing
* Label distribution visualization
* Correlation heatmap between toxicity labels
* Comment length distribution

---

## ⚙️ Text Preprocessing

The following preprocessing steps were applied:

* Convert text to lowercase
* Remove URLs
* Remove HTML tags
* Remove punctuation
* Remove numbers
* Remove stopwords
* Lemmatization using WordNet
* Remove extra whitespace

---

## 🧠 Feature Engineering

Text was converted into numerical features using **TF-IDF Vectorization** with:

* Maximum Features: 30,000
* N-grams: (1,2)
* Minimum Document Frequency: 5

---

## 🤖 Models Implemented

### 1. Base Linear SVM

* One-vs-Rest Classification
* LinearSVC

### 2. Balanced Linear SVM

* LinearSVC
* class_weight="balanced"

### 3. Hyperparameter Tuned SVM

GridSearchCV was used to optimize:

* C
* Loss Function (hinge / squared_hinge)

Best Parameters:

```python
{'estimator__C': 1,
 'estimator__loss': 'hinge'}
```

---

## 📈 Model Performance

| Model            |   Accuracy | Micro F1 | Macro F1 | Hamming Loss |
| :--------------- | ---------: | -------: | -------: | -----------: |
| Base SVM         | **91.86%** | **0.71** |     0.54 |   **0.0185** |
| Balanced SVM     |     87.78% |     0.68 |     0.56 |       0.0279 |
| GridSearchCV SVM |     87.53% |     0.68 | **0.57** |       0.0286 |

### Key Findings

* Base SVM achieved the highest overall accuracy and lowest Hamming Loss.
* Applying class balancing significantly improved recall for minority classes such as **Threat**, **Identity Hate**, and **Severe Toxic**.
* Hyperparameter tuning further improved the Macro F1-score by selecting the optimal loss function (`hinge`).

---

## 💬 Example Prediction

Input:

```text
I will kill you
```

Predicted Labels:

```text
Toxic          : Yes
Threat         : Yes
Severe Toxic   : No
Obscene        : No
Insult         : No
Identity Hate  : No
```

---

## 🚀 Future Improvements

* Deep Learning using LSTM or BiLSTM
* Transformer models (BERT/RoBERTa)
* Explainable AI with SHAP/LIME
* Flask/FastAPI deployment
* Docker containerization
* Cloud deployment on AWS

---

## 📁 Project Structure

```text
Cyberbullying-Detection/
│
├── Cyberbullying-Detection.ipynb
├── train.csv.gz
├── requirements.txt
├── README.md
```

---

## ⭐ Skills Demonstrated

* Natural Language Processing
* Multi-label Classification
* Text Preprocessing
* TF-IDF Feature Engineering
* Support Vector Machines
* Hyperparameter Tuning
* Model Evaluation
* Exploratory Data Analysis
* Python for Machine Learning
