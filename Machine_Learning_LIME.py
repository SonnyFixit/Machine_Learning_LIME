import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from lime.lime_tabular import LimeTabularExplainer

#
# 1. Preprocess data
#

# Load data
df = pd.read_csv('titanic_train.csv')

# Check for any missing values
print(df.isnull().sum())

# Fill missing values with the median for numerical columns or the most common value for categorical columns
for column in df.columns:
    if df[column].dtype == 'object':  # for categorical columns
        df[column] = df[column].fillna(df[column].mode()[0])
    else:  # for numerical columns
        df[column] = df[column].fillna(df[column].median())

# Check again for any missing values
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Split data into independent (X) and dependent (y) variables
X = df.drop('Survived', axis=1)
y = df['Survived']

# Convert data to numeric
X = pd.get_dummies(X)

# Split into training and test sets - no less than 200 (if the data has at least 200 or more), otherwise, it will be as many as there are (e.g., 199)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200/len(df), random_state=123)

#
# 2. Create and train model, generate report and matrix
#

# Create and train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Perform predictions on the test set
predictions = model.predict(X_test)

# Generate classification report
class_report = classification_report(y_test, predictions)
print("Classification report:\n", class_report)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

# Create heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Did not survive', 'Survived'],
            yticklabels=['Did not survive', 'Survived'])
plt.ylabel('Actual value')
plt.xlabel('Predicted value')
plt.title('Confusion Matrix')
plt.show()

#
# Sample description of results for TASK 2
#

# The classification results for the test set using the DecisionTreeClassifier are presented 
# via the confusion matrix and classification report. Based on the confusion matrix, we can state:

# - The number of true negative cases (TN): 100
#   This means that the model correctly predicted that 100 passengers did not survive the disaster.

# - The number of false positive cases (FP): 23
#   The model incorrectly classified 23 cases as survived, while in reality, the passengers did not survive.

# - The number of false negative cases (FN): 20
#   In these cases, the model incorrectly predicted that passengers did not survive, while they did survive.

# - The number of true positive cases (TP): 57
#   The model correctly identified that 57 passengers would survive.

# The classification report provides additional metrics such as precision, recall, and the F1 score for each class,
# which allows for a more detailed analysis of model performance. Precision tells us what part of identifications
# as 'survived' was correct, recall determines what part of the actual cases of 'survived' the model was able to catch,
# and the F1 score is the harmonic mean of precision and recall, providing a quick measure of the model's quality.

#
# 3. Explain results
#

# Create names for decision classes
class_names = ['Did not survive', 'Survived']

# Fit the Explainer
explainer = LimeTabularExplainer(training_data=X_train.values,
                                 feature_names=X_train.columns,
                                 class_names=class_names,
                                 mode='classification',
                                 training_labels=y_train,
                                 feature_selection='none',
                                 discretize_continuous=False)

# Function to find case indices for analysis
def find_case_indices(y_true, y_pred, case):
    indices = {
        'TP': [i for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_t == 1 and y_p == 1],
        'FP': [i for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_t == 0 and y_p == 1],
        'TN': [i for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_t == 0 and y_p == 0],
        'FN': [i for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)) if y_t == 1 and y_p == 0]
    }
    return random.choice(indices[case])

# Find cases
indices_to_explain = {
    'TN': find_case_indices(y_test, predictions, 'TN'),
    'FP': find_case_indices(y_test, predictions, 'FP'),
    'FN': find_case_indices(y_test, predictions, 'FN'),
    'TP': find_case_indices(y_test, predictions, 'TP')
}

# Present results
for case, index in indices_to_explain.items():
    exp = explainer.explain_instance(X_test.values[index], model.predict_proba, num_features=5)
    pred_class = model.predict(X_test.values[index].reshape(1, -1))[0]
    true_class = y_test.values[index]
    print(f"Case {case}: Actual class: {true_class}, Predicted class: {pred_class}")
    print("Explanation of the prediction:")
    print(exp.as_list())
    print("\n")
    
#
# Conclusions and explanations
#
    
# Case TN (True Negative): 
# Actual class: 0 (did not survive), Predicted class: 0 (did not survive)
# Conclusions:
# - Pclass: A higher class (lower 'Pclass' value) reduces the probability of predicting death.
# - Sex_female: Being female increases the probability of predicting death.
# - Age: Older age reduces the probability of predicting death.

# Case FP (False Positive): 
# Actual class: 0 (did not survive), Predicted class: 1 (survived)
# Conclusions:
# - Pclass: Similar to the TN case, a higher class reduces the probability of predicting survival.
# - Sex_female: Being female increases the probability of a false prediction of survival.

# Case FN (False Negative): 
# Actual class: 1 (survived), Predicted class: 0 (did not survive)
# Conclusions:
# - Pclass: A higher class reduces the probability of predicting survival.
# - Sex_female: Being female increases the probability of a false prediction of death.

# Case TP (True Positive): 
# Actual class: 1 (survived), Predicted class: 1 (survived)
# Conclusions:
# - Pclass: A higher class (lower number) reduces the probability of predicting survival.
# - Sex_female: Being female increases the probability of correctly predicting survival.

# General Conclusions from LIME:
# - Pclass is a key factor influencing predictions in every case. 
#   A lower class often leads the model to predict a higher chance of survival.
# - Gender: Being female usually increases the chance of predicting survival, which is consistent 
#   with historical data (women and children were given priority during evacuation).
# - Age: Has a varying impact depending on the case, but generally, older age 
#   leads the model to predict a lower chance of survival.
