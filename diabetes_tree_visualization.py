import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from IPython.display import Image
import pydotplus
import io

# Load the data
print("Loading data...")
data = pd.read_csv("data.csv")

# Prepare features and target
X = data.drop('Diabetes_012', axis=1)
y = data['Diabetes_012']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a more detailed decision tree for visualization purposes
# Using a smaller depth for better visualization
viz_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
viz_tree.fit(X_train, y_train)

# Feature names for better readability
feature_names = [
    'High BP', 'High Cholesterol', 'Cholesterol Check', 'BMI', 
    'Smoker', 'Stroke', 'Heart Disease', 'Physical Activity',
    'Fruits', 'Vegetables', 'Heavy Alcohol', 'Healthcare',
    'No Doctor (Cost)', 'General Health', 'Mental Health', 'Physical Health',
    'Difficulty Walking', 'Sex', 'Age', 'Education', 'Income'
]

# Class names
class_names = ['No Diabetes', 'Prediabetes', 'Diabetes']

# Create dot data for visualization
dot_data = export_graphviz(
    viz_tree,
    out_file=None,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    special_characters=True,
    proportion=True
)

# Create graph from dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Color settings for nodes
colors = ['lightblue', 'lightyellow', 'lightpink']
edges = graph.get_edge_list()
nodes = graph.get_node_list()

for node in nodes:
    if node.get_name() not in ('node', 'edge'):
        values = viz_tree.tree_.value[int(node.get_name())][0]
        # Color based on majority class
        node_class = np.argmax(values)
        if node_class < len(colors):
            node.set_fillcolor(colors[node_class])

# Save the visualization
graph.write_png("diabetes_decision_tree_visualization.png")
print("Decision tree visualization saved as 'diabetes_decision_tree_visualization.png'")

# Create a function to trace a sample path through the tree
def trace_decision_path(sample_data):
    """Trace the decision path for a single sample"""
    # Convert sample to DataFrame
    if isinstance(sample_data, dict):
        sample = pd.DataFrame([sample_data])
    else:
        sample = pd.DataFrame([sample_data], columns=X.columns)
    
    # Get the decision path
    path = viz_tree.decision_path(sample)
    
    # Get the nodes in the path
    node_indices = path.indices
    
    # Trace the path
    print("\nDecision Path Trace:")
    tree = viz_tree.tree_
    
    for i, node_id in enumerate(node_indices):
        # Check if it's a leaf node
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node
            class_counts = tree.value[node_id][0]
            total = class_counts.sum()
            class_probs = class_counts / total
            
            print(f"Step {i+1}: LEAF NODE")
            for j, class_name in enumerate(class_names):
                print(f"   - {class_name}: {class_probs[j]:.2%}")
            
            # Get the prediction
            prediction = np.argmax(class_counts)
            print(f"   → Final Diagnosis: {class_names[prediction]}")
        else:
            # Decision node
            feature = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            value = sample.iloc[0, tree.feature[node_id]]
            
            if value <= threshold:
                decision = f"{feature} ≤ {threshold:.2f} is TRUE (actual value: {value:.2f})"
                next_node = tree.children_left[node_id]
            else:
                decision = f"{feature} ≤ {threshold:.2f} is FALSE (actual value: {value:.2f})"
                next_node = tree.children_right[node_id]
                
            print(f"Step {i+1}: {decision}")

# Sample patient data
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

# Trace the path for the sample patient
print("\n=== Sample Patient Decision Path ===")
trace_decision_path(sample_patient)

# Create a function to generate a text-based decision tree visualization
def print_tree_structure(tree, feature_names, class_names, node_id=0, depth=0):
    """Print a text representation of the decision tree"""
    indent = "  " * depth
    
    if tree.children_left[node_id] == tree.children_right[node_id]:  # Leaf node
        class_counts = tree.value[node_id][0]
        class_probs = class_counts / class_counts.sum()
        max_class = np.argmax(class_counts)
        print(f"{indent}└─ PREDICT: {class_names[max_class]} ", end="")
        print(f"[{', '.join([f'{class_names[i]}: {p:.2%}' for i, p in enumerate(class_probs)])}]")
        return
    
    # Decision node
    feature = feature_names[tree.feature[node_id]]
    threshold = tree.threshold[node_id]
    
    print(f"{indent}├─ {feature} ≤ {threshold:.2f}?")
    
    # Left branch (True)
    print(f"{indent}│  ├─ True:")
    print_tree_structure(tree, feature_names, class_names, tree.children_left[node_id], depth + 2)
    
    # Right branch (False)
    print(f"{indent}│  └─ False:")
    print_tree_structure(tree, feature_names, class_names, tree.children_right[node_id], depth + 2)

# Print text representation of the tree (limited to a reasonable depth)
print("\n=== Text Representation of Decision Tree ===")
print_tree_structure(viz_tree.tree_, feature_names, class_names)

# Create a function to generate a simple interactive diagnostic tool
def text_based_diagnosis():
    """Simple text-based diagnostic tool that uses the decision tree directly"""
    print("\n=== Diabetes Risk Assessment Tool ===")
    
    # Get key features based on our visualization tree
    key_features = {
        'BMI': float(input("BMI: ")),
        'GenHlth': float(input("General Health (1-5, 1=excellent, 5=poor): ")),
        'Age': float(input("Age Category (1-13, 1=18-24, 13=80+ years): ")),
        'HighBP': float(input("High Blood Pressure (1 for Yes, 0 for No): ")),
        'HighChol': float(input("High Cholesterol (1 for Yes, 0 for No): "))
    }
    
    # Fill in remaining features with default values
    full_patient = {
        'HighBP': key_features['HighBP'],
        'HighChol': key_features['HighChol'],
        'CholCheck': 1.0,  # Assume cholesterol has been checked
        'BMI': key_features['BMI'],
        'Smoker': 0.0,
        'Stroke': 0.0,
        'HeartDiseaseorAttack': 0.0,
        'PhysActivity': 1.0,
        'Fruits': 1.0,
        'Veggies': 1.0,
        'HvyAlcoholConsump': 0.0,
        'AnyHealthcare': 1.0,
        'NoDocbcCost': 0.0,
        'GenHlth': key_features['GenHlth'],
        'MentHlth': 0.0,
        'PhysHlth': 0.0,
        'DiffWalk': 0.0,
        'Sex': 0.0,
        'Age': key_features['Age'],
        'Education': 4.0,
        'Income': 5.0
    }
    
    # Convert to DataFrame
    patient_df = pd.DataFrame([full_patient])
    
    # Make prediction
    prediction = viz_tree.predict(patient_df)[0]
    proba = viz_tree.predict_proba(patient_df)[0]
    
    print("\n=== Assessment Result ===")
    print(f"Diagnosis: {class_names[int(prediction)]}")
    print(f"Confidence: {max(proba)*100:.2f}%")
    
    # Trace the path
    trace_decision_path(full_patient)
    
    return class_names[int(prediction)]

# Uncomment to run the interactive tool
# text_based_diagnosis()

print("\nAll visualizations and tools have been created successfully!")