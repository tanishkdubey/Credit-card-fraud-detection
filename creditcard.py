import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import xgboost as xgb


df = pd.read_csv('../data/creditcard.csv')

# print(df.info())
# print(df.describe())

# print(df['Class'].value_counts(normalize=True))

# plt.figure(figsize=(8,6))
# sns.countplot(x='Class' , data = df)
# plt.title('Class Distribution')
# plt.show()

# plt.figure(figsize=(10,6))
# sns.histplot(df[df['Class']==0]['Amount'], bins=50 , color='blue', label ='Legitimate')
# sns.histplot(df[df['Class']==1]['Amount'], bins=50 , color='red', label ='Fraud')
# plt.yscale('log')
# plt.legend()
# plt.show()

X = df.drop('Class' , axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_smote , y_train_smote = smote.fit_resample(X_train, y_train)

scaler = RobustScaler()
X_train_smote[['Amount','Time']] = scaler.fit_transform(X_train_smote[['Amount','Time']])
X_test[['Amount','Time']] = scaler.transform(X_test[['Amount','Time']])

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
xgb_clf.fit(X_train_smote, y_train_smote)

y_pred_xgb = xgb_clf.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC:", roc_auc_score(y_test, y_pred_xgb))

y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_xgb)
ap_score = average_precision_score(y_test, y_proba_xgb)

plt.figure(figsize=(10,6))
plt.plot(recall, precision, label=f'XGBoost (AP={ap_score:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()