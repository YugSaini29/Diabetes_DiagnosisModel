import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load the data
data = pd.read_csv("data.csv")

# Exploratory Data Analysis
print("Dataset Info:")
print(f"Shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head(5))

print("\nData Description:")
print(data.describe())

print("\nCheck for missing values:")
print(data.isnull().sum())

# Define feature columns and target variable
X = data.drop('Diabetes_012', axis=1) # removing the "Diabetes_012" as that is the data we are going to predict.
y = data['Diabetes_012'] # y will be our targetted data that we are going to predict.

# Split the data into training and testing sets
# using 70-30 ratio. 70 for training the model and 30 for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #random_state randomises the data 

print(f"\nTraining set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Create a decision tree classifier
# Starting with a simple model
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
accuracy = accuracy_score(y_test, y_pred) # checking accuracy of our predicted values by comparing them with y_test values
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:") # checking where our model went wrong or "confused" 
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Prediabetes', 'Diabetes'], rounded=True)
plt.title('Decision Tree for Diabetes Diagnosis')
plt.tight_layout()
plt.savefig('decision_tree.png')

# Getting feature importances
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Important Features for Diabetes Diagnosis')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Export the decision tree rules as text
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)

# Create a function to navigate the decision tree (simulate traversal)
def navigate_decision_tree(patient_data):
    """
    Function to navigate through the decision tree for a new patient
    
    Args:
        patient_data: A dictionary with feature names as keys and patient values as values
        
    Returns:
        The predicted diagnosis and the path taken
    """
    # Convert patient data to DataFrame (single row)
    patient_df = pd.DataFrame([patient_data])
    
    # Make prediction
    prediction = dt_classifier.predict(patient_df)[0]
    
    # Get prediction probability
    proba = dt_classifier.predict_proba(patient_df)[0]
    
    # Map prediction to diagnosis
    diagnosis_map = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
    diagnosis = diagnosis_map[prediction]
    
    # Get the decision path
    path = dt_classifier.decision_path(patient_df)
    
    # Get the nodes in the decision path
    node_indices = path.indices
    
    # Print the decision path
    print("\nDecision Path Traversal:")
    tree = dt_classifier.tree_
    feature_names = X.columns
    
    path_description = []
    for node_id in node_indices:
        # Check if the node is a leaf node
        if tree.children_left[node_id] == tree.children_right[node_id]:
            class_counts = tree.value[node_id][0]
            class_probs = class_counts / np.sum(class_counts)
            leaf_info = f"Leaf node reached with class probabilities: {class_probs}"
            path_description.append(leaf_info)
        else:
            # It's a decision node
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            patient_value = patient_df[feature].values[0]
            decision = "≤" if patient_value <= threshold else ">"
            node_info = f"Check if {feature} {decision} {threshold:.2f} (Patient's value: {patient_value:.2f})"
            path_description.append(node_info)
    
    result = {
        "diagnosis": diagnosis,
        "confidence": float(max(proba) * 100),
        "path": path_description
    }
    
    return result

# Example usage: navigate the decision tree for a sample patient
print("\n--- Decision Tree Navigation Example ---")
# Sample patient data (you can modify these values)
sample_patient = {
    'HighBP': 1.0,
    'HighChol': 1.0,
    'CholCheck': 1.0,
    'BMI': 32.0,
    'Smoker': 0.0,
    'Stroke': 0.0,
    'HeartDiseaseorAttack': 0.0,
    'PhysActivity': 0.0,
    'Fruits': 0.0,
    'Veggies': 1.0,
    'HvyAlcoholConsump': 0.0,
    'AnyHealthcare': 1.0,
    'NoDocbcCost': 0.0,
    'GenHlth': 4.0,
    'MentHlth': 5.0,
    'PhysHlth': 10.0,
    'DiffWalk': 0.0,
    'Sex': 1.0,
    'Age': 9.0,
    'Education': 4.0,
    'Income': 5.0
}

result = navigate_decision_tree(sample_patient)
print(f"\nDiagnosis: {result['diagnosis']}")
print(f"Confidence: {result['confidence']:.2f}%")
print("\nPath Taken:")
for step in result['path']:
    print(f"- {step}")

# Create a simple interactive diagnostic tool
def interactive_diabetes_diagnosis():
    # """
    # Interactive function to input patient details and get diagnosis
    # """
    print("\n=== Diabetes Diagnosis System ===")
    print("Please enter the following details for the patient:")
    
    patient_data = {}
    
    # Collect input for each feature
    patient_data['HighBP'] = float(input("High Blood Pressure (1 for Yes, 0 for No): "))
    patient_data['HighChol'] = float(input("High Cholesterol (1 for Yes, 0 for No): "))
    patient_data['CholCheck'] = float(input("Cholesterol Check in 5 years (1 for Yes, 0 for No): "))
    patient_data['BMI'] = float(input("BMI: "))
    patient_data['Smoker'] = float(input("Smoker (1 for Yes, 0 for No): "))
    patient_data['Stroke'] = float(input("Ever had Stroke (1 for Yes, 0 for No): "))
    patient_data['HeartDiseaseorAttack'] = float(input("Heart Disease/Attack (1 for Yes, 0 for No): "))
    patient_data['PhysActivity'] = float(input("Physical Activity (1 for Yes, 0 for No): "))
    patient_data['Fruits'] = float(input("Consumes Fruits (1 for Yes, 0 for No): "))
    patient_data['Veggies'] = float(input("Consumes Vegetables (1 for Yes, 0 for No): "))
    patient_data['HvyAlcoholConsump'] = float(input("Heavy Alcohol Consumption (1 for Yes, 0 for No): "))
    patient_data['AnyHealthcare'] = float(input("Has Healthcare (1 for Yes, 0 for No): "))
    patient_data['NoDocbcCost'] = float(input("No Doctor due to Cost (1 for Yes, 0 for No): "))
    patient_data['GenHlth'] = float(input("General Health (1-5, 1=excellent, 5=poor): "))
    patient_data['MentHlth'] = float(input("Mental Health (0-30 days, 0=excellent, 30=poor): "))
    patient_data['PhysHlth'] = float(input("Physical Health (0-30 days, 0=excellent, 30=poor): "))
    patient_data['DiffWalk'] = float(input("Difficulty Walking (1 for Yes, 0 for No): "))
    patient_data['Sex'] = float(input("Sex (1 for Male, 0 for Female): "))
    patient_data['Age'] = float(input("Age Category (1-13, 1=18-24, 13=80+ years): "))
    patient_data['Education'] = float(input("Education Level (1-6, 1=lowest, 6=highest): "))
    patient_data['Income'] = float(input("Income Level (1-8, 1=lowest, 8=highest): "))
    
    # Get diagnosis
    result = navigate_decision_tree(patient_data)
    
    print("\n=== Diagnosis Results ===")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print("\nDecision Path:")
    for step in result['path']:
        print(f"- {step}")
    
    return result

# Uncomment the line below to run the interactive tool
# interactive_diabetes_diagnosis()

# Save the model for future use
import pickle
with open('diabetes_decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt_classifier, f)

print("\nModel saved as 'diabetes_decision_tree_model.pkl'")