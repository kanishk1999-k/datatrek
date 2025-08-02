#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


# In[2]:


# Set plot style for better visuals
plt.style.use('seaborn-v0_8-whitegrid')

# Load the dataset
# Assumes the dataset is in the working directory
df = pd.read_csv('Employee.csv')
print(df.shape)  # Output: (1470, 35)
print(df.columns)


# In[3]:


# Display the first few rows to understand the data
print("First 5 rows of the dataset:")
print(df.head())


# In[4]:


# Check the shape of the dataset (rows, columns)
print("\nDataset shape:", df.shape)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())


# In[5]:


# Check data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())


# In[6]:


# Check unique values for categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nUnique values in categorical columns:")
for col in categorical_cols:
    print(f"{col}: {df[col].unique()}")


# In[7]:


# Visualize the distribution of the target variable (Attrition)
plt.figure(figsize=(8, 6))
sns.countplot(x='Attrition', data=df)
plt.title('Distribution of Attrition')
plt.xlabel('Attrition (Yes/No)')
plt.ylabel('Count')
plt.show()


# In[8]:


# Correlation matrix for numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
plt.figure(figsize=(12, 10))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()


# In[9]:


# Visualize Attrition by Department
plt.figure(figsize=(10, 6))
sns.countplot(x='Department', hue='Attrition', data=df)
plt.title('Attrition by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Attrition')
plt.show()


# In[10]:


#Attrition by monthly income
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.title('Income vs. Attrition')
plt.show()


# In[11]:


# Visualize Attrition by Job Satisfaction
plt.figure(figsize=(10, 6))
sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
plt.title('Attrition by Job Satisfaction')
plt.xlabel('Job Satisfaction (1=Low, 4=Very High)')
plt.ylabel('Count')
plt.legend(title='Attrition')
plt.show()


# In[12]:


# Drop irrelevant columns that don't contribute to prediction
df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)


# In[13]:


# Encode the target variable (Attrition: Yes=1, No=0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


# In[14]:


# Separate features (X) and target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']


# In[15]:


# Define categorical and numerical columns for preprocessing
categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numerical_cols = [col for col in X.columns if col not in categorical_cols]


# In[16]:


# Split the data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Define preprocessing steps
# Scale numerical features and one-hot encode categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])


# In[18]:


# Linear Regression Model (not ideal for classification, included to invalidate)
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# In[19]:


# Train Linear Regression
lr_pipeline.fit(X_train, y_train)


# In[20]:


# Predict and convert to binary classes using 0.5 threshold
y_pred_lr = lr_pipeline.predict(X_test)
y_pred_lr_class = (y_pred_lr > 0.5).astype(int)


# In[21]:


# Evaluate Linear Regression
print("\nLinear Regression Performance:")
print(classification_report(y_test, y_pred_lr_class))
print("Accuracy:", accuracy_score(y_test, y_pred_lr_class))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_lr))


# In[22]:


# Logistic Regression Model (better suited for classification)
logreg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])


# In[23]:


# Train Logistic Regression
logreg_pipeline.fit(X_train, y_train)


# In[24]:


# Predict
y_pred_logreg = logreg_pipeline.predict(X_test)
y_pred_logreg_proba = logreg_pipeline.predict_proba(X_test)[:, 1]


# In[25]:


# Evaluate Logistic Regression
print("\nLogistic Regression Performance:")
print(classification_report(y_test, y_pred_logreg))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_logreg_proba))


# In[26]:


# K-Nearest Neighbors (KNN) Model
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsClassifier())
])


# In[27]:


# Train KNN
knn_pipeline.fit(X_train, y_train)


# In[28]:


# Predict
y_pred_knn = knn_pipeline.predict(X_test)
y_pred_knn_proba = knn_pipeline.predict_proba(X_test)[:, 1]


# In[29]:


# Evaluate KNN
print("\nKNN Performance:")
print(classification_report(y_test, y_pred_knn))
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_knn_proba))


# In[30]:


# Random Forest Classifier Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Train Random Forest
rf_pipeline.fit(X_train, y_train)


# In[31]:


# Predict
y_pred_rf = rf_pipeline.predict(X_test)
y_pred_rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

# Evaluate Random Forest
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_rf_proba))


# In[32]:


# Compare model performance
model_performance = {
    'Model': ['Linear Regression', 'Logistic Regression', 'KNN', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr_class),
        accuracy_score(y_test, y_pred_logreg),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_rf)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_lr_class),
        f1_score(y_test, y_pred_logreg),
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_rf)
    ],
    'ROC-AUC': [
        roc_auc_score(y_test, y_pred_lr),
        roc_auc_score(y_test, y_pred_logreg_proba),
        roc_auc_score(y_test, y_pred_knn_proba),
        roc_auc_score(y_test, y_pred_rf_proba)
    ]
}


# In[33]:


# Display performance comparison
performance_df = pd.DataFrame(model_performance)
print("\nModel Performance Comparison:")
print(performance_df)


# In[34]:


from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, logreg_pipeline.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Logistic Regression')
plt.show()


# In[35]:


# Extract feature names and coefficients (using existing lr_pipeline)
feature_names = logreg_pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = logreg_pipeline.named_steps['model'].coef_[0]

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort by absolute coefficient value
feature_importance_df['Abs_Coefficient'] = feature_importance_df['Coefficient'].abs()
feature_importance_df = feature_importance_df.sort_values(by='Abs_Coefficient', ascending=False)

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importance (Logistic Regression)')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.show()

# Business Insights
print("\nBusiness Insights:")
print("1. **Key Drivers of Attrition**: Features with large positive coefficients (e.g., overtime, low job satisfaction) increase attrition risk.")
print("2. **Retention Strategies**: Enhance factors with large negative coefficients (e.g., income, tenure) to improve retention.")
print("3. **Targeted Interventions**: Use model insights to focus on high-risk groups (e.g., specific departments or roles).")
print("4. **Model Selection**: Logistic Regression is preferred for its superior performance (Accuracy: 0.895, F1-Score: 0.537) and interpretability.")


# In[38]:


# # Step 1: Install and import SMOTE (if not already installed, run !pip install imbalanced-learn in a cell)
# # This brings in tools for resampling. imblearn.pipeline allows SMOTE to fit inside your workflow, applying it only to training data after preprocessing (to avoid data leakage).
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline
# from imblearn.over_sampling import BorderlineSMOTE

# # Step 2: Create SMOTE-enabled pipelines for each model
# # Reuse your existing preprocessor (handles scaling and encoding). Add SMOTE (random_state=42 for reproducibility—it generates consistent synthetic samples).
# # SMOTE fits on preprocessed training data, creates balanced samples, then trains the model. For example, if minority class is 20% of train, SMOTE upsamples it to 50% by default.
# logreg_pipeline_smote = ImbPipeline([
#     ('preprocessor', preprocessor),
#     ('smote', BorderlineSMOTE(random_state=42)),
#     ('model', LogisticRegression(random_state=42))
# ])

# knn_pipeline_smote = ImbPipeline([
#     ('preprocessor', preprocessor),
#     ('smote', BorderlineSMOTE(random_state=42)),
#     ('model', KNeighborsClassifier())
# ])

# rf_pipeline_smote = ImbPipeline([
#     ('preprocessor', preprocessor),
#     ('smote', BorderlineSMOTE(random_state=42)),
#     ('model', RandomForestClassifier(random_state=42))
# ])

# # Step 3: Train the SMOTE pipelines on the original train data
# # Fitting preprocesses X_train (turns categories to numbers, scales), applies SMOTE to balance y_train, then trains the model. No changes to test data—keeps evaluation fair.
# logreg_pipeline_smote.fit(X_train, y_train)
# knn_pipeline_smote.fit(X_train, y_train)
# rf_pipeline_smote.fit(X_train, y_train)

# # Step 4: Make predictions on test data
# # Pipelines automatically preprocess X_test before predicting, ensuring consistency.
# y_pred_logreg_smote = logreg_pipeline_smote.predict(X_test)
# y_pred_knn_smote = knn_pipeline_smote.predict(X_test)
# y_pred_rf_smote = rf_pipeline_smote.predict(X_test)

# y_prob_logreg_smote = logreg_pipeline_smote.predict_proba(X_test)[:, 1]
# y_prob_knn_smote = knn_pipeline_smote.predict_proba(X_test)[:, 1]
# y_prob_rf_smote = rf_pipeline_smote.predict_proba(X_test)[:, 1]

# # Step 5: Define a function to compute metrics (reuse your existing logic for consistency)
# # This calculates accuracy (overall correct), precision (true 'Yes' among predicted 'Yes'), recall (caught 'Yes' cases), F1 (harmonic mean of precision/recall), ROC-AUC (how well it separates classes).
# # For imbalanced data, focus on recall and F1—accuracy can mislead if it just predicts 'No' often.
# def evaluate_model(y_true, y_pred, y_prob):
#     return {
#         'Accuracy': accuracy_score(y_true, y_pred),
#         'Precision': precision_score(y_true, y_pred),
#         'Recall': recall_score(y_true, y_pred),
#         'F1-Score': f1_score(y_true, y_pred),
#         'ROC-AUC': roc_auc_score(y_true, y_prob)
#     }

# # Compute metrics for SMOTE models
# metrics_logreg_smote = evaluate_model(y_test, y_pred_logreg_smote, y_prob_logreg_smote)
# metrics_knn_smote = evaluate_model(y_test, y_pred_knn_smote, y_prob_knn_smote)
# metrics_rf_smote = evaluate_model(y_test, y_pred_rf_smote, y_prob_rf_smote)

# # Step 6: Compare metrics in a table (without SMOTE vs. with SMOTE)
# # Pull your original metrics (assuming you stored them as metrics_logreg, metrics_knn, metrics_rf from earlier code).
# # This DataFrame shows changes—e.g., recall might rise (better at spotting attrition), but precision could drop (more false positives). Think: Does the trade-off help business goals, like retaining talent?
# comparison_df = pd.DataFrame({
#     'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],

#     'Logistic Regression (SMOTE)': [metrics_logreg_smote['Accuracy'], metrics_logreg_smote['Precision'], metrics_logreg_smote['Recall'], metrics_logreg_smote['F1-Score'], metrics_logreg_smote['ROC-AUC']],
    
#     'KNN (SMOTE)': [metrics_knn_smote['Accuracy'], metrics_knn_smote['Precision'], metrics_knn_smote['Recall'], metrics_knn_smote['F1-Score'], metrics_knn_smote['ROC-AUC']],
    
#     'Random Forest (SMOTE)': [metrics_rf_smote['Accuracy'], metrics_rf_smote['Precision'], metrics_rf_smote['Recall'], metrics_rf_smote['F1-Score'], metrics_rf_smote['ROC-AUC']]
# })

# print("\nPerformance Comparison (No SMOTE vs. SMOTE):\n")
# print(comparison_df)

# # Optional: Visualize changes (bar plot for F1-Score across models)
# # This helps spot patterns—e.g., SMOTE often boosts F1 for imbalanced classes, like predicting rare diseases in medicine.
# plt.figure(figsize=(10, 6))
# sns.barplot(x='Metric', y='value', hue='Model', data=comparison_df.melt(id_vars='Metric', var_name='Model', value_name='value')[comparison_df.melt(id_vars='Metric').Metric == 'F1-Score'])
# plt.title('F1-Score Comparison: No SMOTE vs. SMOTE')
# plt.xticks(rotation=45)
# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




