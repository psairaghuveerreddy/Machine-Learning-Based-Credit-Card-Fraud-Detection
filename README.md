---
# 💳 Machine Learning–Based Credit Card Fraud Detection
---
🚀 Imbalanced Data Classification | Financial Security Project
---
## 📘 Project Overview

This project focuses on detecting fraudulent credit card transactions using Machine Learning classification techniques.
Due to the highly imbalanced nature of fraud datasets, traditional accuracy-based evaluation is insufficient. 
Hence, this project emphasizes precision, recall, F1-score, and ROC-AUC to build a reliable fraud detection system.
The trained models analyze transaction patterns and classify each transaction as legitimate or fraudulent, helping financial institutions prevent losses and enhance transaction security.
---
## 🎯 Objectives
- Build a machine learning–based fraud detection system
- Handle highly imbalanced transaction data
- Compare multiple classification algorithms
- Optimize fraud detection using suitable evaluation metrics
- Identify the best-performing model for real-world usage
---
## 🧩 Dataset Information

Dataset: Credit Card Transactions Dataset
Transactions: European cardholder transactions
Features: Numerical & anonymized (for privacy protection)
Target Variable:
0 → Legitimate Transaction
1 → Fraudulent Transaction
Challenge: Severe class imbalance (fraud cases < 1%)
---
## ⚙️ System Workflow

Transaction Dataset
   ↓
Data Preprocessing & Cleaning
   ↓
Feature Scaling & Class Balancing
   ↓
Exploratory Data Analysis (EDA)
   ↓
Model Training
   ↓
Model Evaluation & Comparison
   ↓
Fraud Prediction
---
🧠 Machine Learning Models Used

|Model|Purpose|
|Logistic Regression|Baseline fraud classification|
|Decision|Tree	Rule-based fraud detection|
|Random Forest|Ensemble learning for better accuracy|
|Support Vector Machine (SVM)|Margin-based classification|
|Gradient Boosting (optional)|Improved fraud detection|
---
## 📊 Evaluation Metrics

Due to class imbalance, the following metrics are used:
Precision
Recall
F1-Score
Confusion Matrix
ROC-AUC Curve
⚠️ Accuracy alone is not reliable for fraud detection problems.
---
## 📈 Model Performance (Best Model Example)

|Metric| Value|
|Accuracy|98%+|
|Precision|High (low false positives)|
|Recall|Improved fraud detection|
|F1-Score|Balanced performance|
|ROC-AUC|Strong class separation|
✅ Random Forest performed best on imbalanced data.
---
## 🧮 Evaluation Methodology

- Split dataset into training & testing sets
- Scale features using standardization
- Train multiple classifiers
- Compare performance using confusion matrix & F1-score
- Select best model for fraud detection
---
```
📂 Project Structure
Credit Card Fraud Detection Using Machine Learning/
│
├── ML___Credit_Card_Fraud_Detection.ipynb
├── README.md
└── requirements.txt
```
---

## ▶️ Running Instructions

1️. **Install Dependencies**: pip install numpy pandas matplotlib seaborn scikit-learn

2️. **Run the Jupyter Notebook**:jupyter notebook ML___Credit_Card_Fraud_Detection.ipynb

3️. **Execute All Cells**:
|Dataset loading|Preprocessing|Model training|Evaluation & comparison|

---

## 📈 Visualizations

✔ Class imbalance visualization
✔ Correlation heatmap
✔ Confusion matrix
✔ ROC-AUC curve
✔ Model comparison plots

---
## ⭐ Key Features

Handles real-world imbalanced data
Multiple ML models comparison
Emphasis on fraud-sensitive metrics
Clear and interpretable results
Suitable for financial security systems
---
## 🚀 Future Enhancements

Apply Deep Learning (LSTM / Autoencoders)
Implement real-time fraud detection
Deploy using Flask / FastAPI
Integrate streaming transaction data
Apply advanced resampling techniques (SMOTE, ADASYN)
---
## 🏁 Conclusion

This project demonstrates a practical machine learning approach to credit card fraud detection.
By handling data imbalance and using appropriate evaluation metrics, the system achieves high fraud detection performance, making it suitable for banking and financial applications.
---
## 👨‍💻 Author

P. Sai Raghuveer Reddy
Department of Artificial Intelligence & Machine Learning
RNS Institute of Technology, Bengaluru
Year: 2025
---
## 🙏 Acknowledgements

Dataset: Public Credit Card Fraud Dataset
Tools: Python, Scikit-learn, NumPy, Pandas, Matplotlib
Guidance: Dr. Mallikarjun H M, Assistant Professor , Department of AIML, RNSIT
---
## 🔑 Keywords

· Credit Card Fraud Detection 
· Machine Learning 
· Imbalanced Data 
· Financial Security 
· Classification 
· Data Mining
---
