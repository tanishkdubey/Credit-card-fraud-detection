📌 Overview

This project implements a machine learning solution to detect fraudulent credit card transactions using the XGBoost algorithm. The dataset is highly imbalanced, with fraud cases representing only 0.17% of all transactions. We address this challenge using SMOTE (Synthetic Minority Over-sampling Technique) for class imbalance handling and RobustScaler for feature scaling.

🚀 Key Features

->Advanced Imbalance Handling: Utilizes SMOTE to synthetically generate minority class samples

->Powerful Algorithm: Implements XGBoost with optimized hyperparameters for fraud detection

->Robust Preprocessing: Uses RobustScaler to handle outliers in 'Amount' and 'Time' features

->Comprehensive Evaluation: Provides multiple evaluation metrics including precision-recall curves and ROC AUC

📊 Performance Metrics

->The model achieves:

->High precision and recall for fraud detection

->ROC AUC score as shown in evaluation

Detailed classification report with precision, recall, and f1-scores

🛠️ Technical Implementation

python
# Key components
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

# Model configuration
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
📈 Results Visualization

The project includes visualizations of:

1)Class distribution before and after SMOTE

2)Transaction amount distributions (log scale)

3)Precision-Recall curves with Average Precision score

💾 Dataset

The dataset comes from Kaggle Credit Card Fraud Detection and contains:

->284,807 transactions

->31 features (28 anonymized + Time, Amount, Class)

->Highly imbalanced classes (492 frauds vs 284,315 legitimate transactions)

� How to Use

1)Clone the repository

2)Install requirements: pip install -r requirements.txt

3)Run the Jupyter notebook or Python script

4)View results and visualizations

📝 Requirements

->Python 3.7+

->pandas, numpy

->scikit-learn, imbalanced-learn

->xgboost

->matplotlib, seaborn

🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements.

📜 License

This project doesn't currently have an explicit license. All rights are reserved by default.

If you'd like to use or contribute to this code:

1)For personal/educational use: Feel free to explore! (Just don't sue me if your credit card actually gets hacked)

2)For commercial use: Contact me to discuss permissions.

3)Want to add a license? Open an issue or PR suggesting one (MIT/Apache/GPL are common choices).
