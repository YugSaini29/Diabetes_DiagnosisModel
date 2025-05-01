import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Diabetes Diagnosis System",
    page_icon="🩺",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4682B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .normal-text {
        font-size: 1rem;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .highlight {
        background-color: #F0F8FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4682B4;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #777;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='main-header'>Diabetes Diagnosis Decision Tree System</h1>", unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Diagnosis Tool", "Model Information", "About"])

# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open('diabetes_decision_tree_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please run the model training script first.")
        return None

# Function to load the data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("data.csv")
        return data
    except FileNotFoundError:
        st.error("Data file not found! Please make sure 'data.csv' is in the current directory.")
        return None

# Load model and data
model = load_model()
data = load_data()

# Home page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to the Diabetes Diagnosis System</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p class='normal-text'>This application uses a decision tree algorithm to predict diabetes risk based on various health factors. The model has been trained on a dataset containing health indicators and diabetes status of individuals.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 class='sub-header'>How to Use This System:</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    1. **Diagnosis Tool**: Enter your health information and receive a diabetes risk assessment
    2. **Model Information**: View details about the decision tree model and its performance
    3. **About**: Learn more about this project and decision trees in medical diagnosis
    """)
    
    if data is not None:
        st.markdown("<h3 class='sub-header'>Dataset Overview:</h3>", unsafe_allow_html=True)
        st.write(f"Total records: {len(data)}")
        
        # Display class distribution
        st.markdown("<h4>Diabetes Distribution in Dataset:</h4>", unsafe_allow_html=True)
        diabetes_counts = data['Diabetes_012'].value_counts()
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(diabetes_counts, labels=['No Diabetes', 'Prediabetes', 'Diabetes'], 
               autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ffcc99', '#ff9999'])
        ax.axis('equal')
        st.pyplot(fig)
        
        # Display a few key features
        st.markdown("<h4>Key Health Indicators:</h4>", unsafe_allow_html=True)
        cols = ['BMI', 'HighBP', 'HighChol', 'GenHlth', 'Age']
        st.write(data[cols].describe())

# Diagnosis Tool page
elif page == "Diagnosis Tool":
    st.markdown("<h2 class='sub-header'>Diabetes Risk Assessment Tool</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p class='normal-text'>Enter your health information below to receive a diabetes risk assessment. The system will use a decision tree algorithm to analyze your risk based on the provided inputs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for input fields
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Personal Information</h3>", unsafe_allow_html=True)
        age_category = st.slider("Age Category", 1, 13, 7, 
                                help="1=18-24, 2=25-29, 3=30-34, 4=35-39, 5=40-44, 6=45-49, 7=50-54, 8=55-59, 9=60-64, 10=65-69, 11=70-74, 12=75-79, 13=80+")
        sex = st.radio("Sex", ["Female", "Male"])
        sex_value = 0.0 if sex == "Female" else 1.0
        
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        education = st.slider("Education Level", 1, 6, 4, 
                             help="1=Never attended school, 2=Elementary, 3=Some high school, 4=High school graduate, 5=Some college or technical school, 6=College graduate")
        income = st.slider("Income Level", 1, 8, 5,
                          help="1=Less than $10k, 2=$10k-$15k, 3=$15k-$20k, 4=$20k-$25k, 5=$25k-$35k, 6=$35k-$50k, 7=$50k-$75k, 8=$75k+")
    
    with col2:
        st.markdown("<h3>Health Indicators</h3>", unsafe_allow_html=True)
        high_bp = st.radio("High Blood Pressure", ["No", "Yes"])
        high_bp_value = 0.0 if high_bp == "No" else 1.0
        
        high_chol = st.radio("High Cholesterol", ["No", "Yes"])
        high_chol_value = 0.0 if high_chol == "No" else 1.0
        
        chol_check = st.radio("Cholesterol Check in 5 years", ["No", "Yes"])
        chol_check_value = 0.0 if chol_check == "No" else 1.0
        
        smoker = st.radio("Smoker", ["No", "Yes"])
        smoker_value = 0.0 if smoker == "No" else 1.0
        
        gen_hlth = st.slider("General Health", 1, 5, 3,
                           help="1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor")
    
    # Additional health factors with default values
    with st.expander("Advanced Health Factors (Optional)"):
        col3, col4 = st.columns(2)
        
        with col3:
            stroke = st.radio("Ever had Stroke", ["No", "Yes"], index=0)
            stroke_value = 0.0 if stroke == "No" else 1.0
            
            heart_disease = st.radio("Heart Disease/Attack", ["No", "Yes"], index=0)
            heart_disease_value = 0.0 if heart_disease == "No" else 1.0
            
            phys_activity = st.radio("Physical Activity in past 30 days", ["No", "Yes"])
            phys_activity_value = 0.0 if phys_activity == "No" else 1.0
            
            fruits = st.radio("Consume Fruits daily", ["No", "Yes"], index=1)
            fruits_value = 0.0 if fruits == "No" else 1.0
            
            veggies = st.radio("Consume Vegetables daily", ["No", "Yes"], index=1)  
            veggies_value = 0.0 if veggies == "No" else 1.0
        
        with col4:
            hvy_alcohol = st.radio("Heavy Alcohol Consumption", ["No", "Yes"], index=0)
            hvy_alcohol_value = 0.0 if hvy_alcohol == "No" else 1.0
            
            healthcare = st.radio("Has Healthcare", ["No", "Yes"], index=1)
            healthcare_value = 0.0 if healthcare == "No" else 1.0
            
            no_doc_cost = st.radio("No Doctor due to Cost", ["No", "Yes"], index=0)
            no_doc_cost_value = 0.0 if no_doc_cost == "No" else 1.0
            
            ment_hlth = st.slider("Days of Poor Mental Health (past 30 days)", 0, 30, 0)
            phys_hlth = st.slider("Days of Poor Physical Health (past 30 days)", 0, 30, 0)
            
            diff_walk = st.radio("Difficulty Walking", ["No", "Yes"], index=0)
            diff_walk_value = 0.0 if diff_walk == "No" else 1.0
    
    # Submit button for diagnosis
    st.markdown("<br>", unsafe_allow_html=True)
    diagnose_button = st.button("Get Diagnosis", type="primary")
    
    if diagnose_button and model is not None:
        # Prepare patient data
        patient_data = {
            'HighBP': high_bp_value,
            'HighChol': high_chol_value,
            'CholCheck': chol_check_value,
            'BMI': bmi,
            'Smoker': smoker_value,
            'Stroke': stroke_value,
            'HeartDiseaseorAttack': heart_disease_value,
            'PhysActivity': phys_activity_value,
            'Fruits': fruits_value,
            'Veggies': veggies_value,
            'HvyAlcoholConsump': hvy_alcohol_value,
            'AnyHealthcare': healthcare_value,
            'NoDocbcCost': no_doc_cost_value,
            'GenHlth': float(gen_hlth),
            'MentHlth': float(ment_hlth),
            'PhysHlth': float(phys_hlth),
            'DiffWalk': diff_walk_value,
            'Sex': sex_value,
            'Age': float(age_category),
            'Education': float(education),
            'Income': float(income)
        }
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])
        
        # Make prediction
        prediction = model.predict(patient_df)[0]
        probabilities = model.predict_proba(patient_df)[0]
        
        # Map prediction to diagnosis
        diagnosis_map = {0.0: "No Diabetes", 1.0: "Prediabetes", 2.0: "Diabetes"}
        diagnosis = diagnosis_map[prediction]
        
        # Get decision path
        path = model.decision_path(patient_df)
        node_indices = path.indices
        
        # Display result with styling
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Diagnosis Result</h2>", unsafe_allow_html=True)
        
        # Different color based on diagnosis
        color = "#5cb85c" if prediction == 0.0 else "#f0ad4e" if prediction == 1.0 else "#d9534f"
        
        st.markdown(f"""
        <div style="background-color: {color}; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
            <h2>Diagnosis: {diagnosis}</h2>
            <h3>Confidence: {probabilities[int(prediction)]:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create three columns for probability visualization
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3>Probability Distribution</h3>", unsafe_allow_html=True)
        
        # Create bar chart of probabilities
        fig, ax = plt.subplots(figsize=(10, 5))
        categories = ['No Diabetes', 'Prediabetes', 'Diabetes']
        colors = ['#5cb85c', '#f0ad4e', '#d9534f']
        
        ax.bar(categories, probabilities, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Diagnosis Probability Distribution')
        
        for i, prob in enumerate(probabilities):
            ax.text(i, prob + 0.05, f"{prob:.1%}", ha='center')
        
        st.pyplot(fig)
        
        # Display decision path
        st.markdown("<h3>Decision Path</h3>", unsafe_allow_html=True)
        st.markdown("<p>This shows how the decision tree arrived at its diagnosis:</p>", unsafe_allow_html=True)
        
        # Create a function to extract the decision path
        tree = model.tree_
        feature_names = list(patient_df.columns)
        
        path_description = []
        for node_id in node_indices:
            # Check if the node is a leaf node
            if tree.children_left[node_id] == tree.children_right[node_id]:
                class_counts = tree.value[node_id][0]
                class_probs = class_counts / np.sum(class_counts)
                leaf_info = f"Final node reached with class probabilities: No Diabetes: {class_probs[0]:.1%}, Prediabetes: {class_probs[1]:.1%}, Diabetes: {class_probs[2]:.1%}"
                path_description.append(leaf_info)
            else:
                # It's a decision node
                feature = feature_names[tree.feature[node_id]]
                threshold = tree.threshold[node_id]
                patient_value = patient_df[feature].values[0]
                decision = "≤" if patient_value <= threshold else ">"
                
                # Improve feature name display
                display_feature = feature.replace("Consump", "Consumption").replace("PhysActivity", "Physical Activity")
                
                node_info = f"Check if {display_feature} {decision} {threshold:.2f} (Your value: {patient_value:.2f})"
                path_description.append(node_info)
        
        # Display path as numbered steps
        for i, step in enumerate(path_description, 1):
            st.markdown(f"**Step {i}:** {step}")
        
        # Risk factors section
        st.markdown("<h3>Key Risk Factors</h3>", unsafe_allow_html=True)
        
        # Identify top risk factors based on feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            top_features = feature_importance.head(5)
            
            # Create horizontal bar chart for feature importance
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
            ax.set_xlabel('Importance')
            ax.set_title('Top 5 Important Features for Diabetes Diagnosis')
            ax.invert_yaxis()  # Display the highest importance at the top
            
            st.pyplot(fig)
            
            # Suggest some recommendations based on modifiable risk factors
            st.markdown("<h3>Recommendations</h3>", unsafe_allow_html=True)
            
            recommendations = []
            
            if bmi > 25:
                recommendations.append("Consider a weight management plan as your BMI is above the healthy range.")
            
            if high_bp_value == 1.0:
                recommendations.append("Monitor and manage your blood pressure with the help of a healthcare professional.")
                
            if high_chol_value == 1.0:
                recommendations.append("Work on lowering your cholesterol levels through diet, exercise, and possibly medication.")
                
            if smoker_value == 1.0:
                recommendations.append("Consider quitting smoking to reduce diabetes risk and improve overall health.")
                
            if phys_activity_value == 0.0:
                recommendations.append("Incorporate regular physical activity into your routine.")
                
            if fruits_value == 0.0 or veggies_value == 0.0:
                recommendations.append("Increase your intake of fruits and vegetables.")
                
            if hvy_alcohol_value == 1.0:
                recommendations.append("Reduce alcohol consumption to improve insulin sensitivity.")
                
            if gen_hlth >= 4:
                recommendations.append("Work with healthcare professionals to improve your general health.")
                
            if len(recommendations) > 0:
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            else:
                st.markdown("Continue maintaining your current healthy lifestyle choices.")
                
            st.markdown("""
            <div class='highlight'>
            <p><strong>Disclaimer:</strong> This tool provides a risk assessment based on statistical analysis and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.</p>
            </div>
            """, unsafe_allow_html=True)

# Model Information page
elif page == "Model Information":
    st.markdown("<h2 class='sub-header'>Decision Tree Model Information</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p class='normal-text'>This page provides technical information about the decision tree model used for diabetes diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if model is not None and data is not None:
        # Split the view into tabs
        tab1, tab2, tab3 = st.tabs(["Model Overview", "Decision Tree Visualization", "Feature Importance"])
        
        with tab1:
            st.markdown("<h3>Model Parameters</h3>", unsafe_allow_html=True)
            
            # Extract and display model parameters
            params = model.get_params()
            st.json(params)
            
            # Display model metrics (if available)
            st.markdown("<h3>Model Performance</h3>", unsafe_allow_html=True)
            
            # Since we don't have test results here, we'll create a placeholder
            # In a real application, you might load these metrics from a file
            st.markdown("""
            Example metrics (placeholder):
            - Accuracy: 85%
            - Precision: 83% 
            - Recall: 81%
            - F1-Score: 82%
            """)
            
            # Display confusion matrix
            st.markdown("<h3>Confusion Matrix Example</h3>", unsafe_allow_html=True)
            
            # Create a sample confusion matrix (for demonstration)
            cm = np.array([[1200, 150, 50], [100, 300, 100], [50, 100, 250]])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                      xticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'],
                      yticklabels=['No Diabetes', 'Prediabetes', 'Diabetes'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Example Confusion Matrix')
            st.pyplot(fig)
            
            # Class distribution
            st.markdown("<h3>Class Distribution</h3>", unsafe_allow_html=True)
            
            diabetes_counts = data['Diabetes_012'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(['No Diabetes', 'Prediabetes', 'Diabetes'], 
                   diabetes_counts.values, 
                   color=['#66b3ff', '#ffcc99', '#ff9999'])
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution in Training Data')
            st.pyplot(fig)
            
        with tab2:
            st.markdown("<h3>Decision Tree Structure</h3>", unsafe_allow_html=True)
            
            # Create a simplified tree visualization
            st.markdown("""
            Below is a visualization of the decision tree. Due to the complexity of the full tree, 
            this is a simplified version with limited depth for better readability.
            """)
            
            # Create a simplified tree for visualization
            from sklearn.tree import DecisionTreeClassifier
            viz_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
            
            # Train on a sample of the data for quicker viz
            if len(data) > 1000:
                sample_data = data.sample(1000, random_state=42)
            else:
                sample_data = data
                
            X_sample = sample_data.drop('Diabetes_012', axis=1)  
            y_sample = sample_data['Diabetes_012']
            viz_tree.fit(X_sample, y_sample)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(15, 10))
            plot_tree(viz_tree, filled=True, feature_names=X_sample.columns, 
                      class_names=['No Diabetes', 'Prediabetes', 'Diabetes'], rounded=True, ax=ax)
            plt.title('Simplified Decision Tree for Diabetes Diagnosis')
            st.pyplot(fig)
            
        with tab3:
            st.markdown("<h3>Feature Importance</h3>", unsafe_allow_html=True)
            
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'Feature': data.drop('Diabetes_012', axis=1).columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
                ax.set_xlabel('Importance')
                ax.set_title('Feature Importance in Decision Tree Model')
                ax.invert_yaxis()  # Display the highest importance at the top
                
                st.pyplot(fig)
                
                # Table of feature importances
                st.markdown("<h4>Feature Importance Table</h4>", unsafe_allow_html=True)
                st.dataframe(feature_importance.style.background_gradient(cmap='Blues'))
                
                # Feature correlations with target
                st.markdown("<h4>Feature Correlation with Diabetes</h4>", unsafe_allow_html=True)
                
                # Calculate correlations
                correlations = data.corr()['Diabetes_012'].drop('Diabetes_012').sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.barh(correlations.index, correlations.values, color='lightcoral')
                ax.set_xlabel('Correlation with Diabetes')
                ax.set_title('Feature Correlation with Diabetes')
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                ax.invert_yaxis()  # Display the highest correlation at the top
                
                st.pyplot(fig)
    else:
        st.error("Model or data not loaded. Please make sure the necessary files are available.")

# About page
elif page == "About":
    st.markdown("<h2 class='sub-header'>About This Project</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='highlight'>
    <p class='normal-text'>This project demonstrates the application of decision trees for medical diagnosis, specifically for diabetes risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Project Overview</h3>", unsafe_allow_html=True)
    st.markdown("""
    This application implements a decision tree algorithm to predict diabetes risk based on various health indicators. The system allows users to:
    
    1. Input their health information
    2. Receive a diabetes risk assessment
    3. View the decision path that led to the diagnosis
    4. Get personalized recommendations based on risk factors
    
    The decision tree model is chosen for its interpretability, which is crucial in medical applications where understanding the reasoning behind a diagnosis is important.
    """)
    
    st.markdown("<h3>How Decision Trees Work in Medical Diagnosis</h3>", unsafe_allow_html=True)
    st.markdown("""
    Decision trees are a popular algorithm in medical diagnosis for several reasons:
    
    - **Interpretability**: Decision trees provide a clear path of reasoning that can be followed and understood by healthcare professionals
    - **Feature Importance**: They identify which health factors are most important for diagnosis
    - **Non-linear Relationships**: They can capture complex relationships between health indicators
    - **Handling Mixed Data Types**: They work well with both numerical (like BMI) and categorical (like Yes/No) health data
    
    In this application, the decision tree examines various health indicators and creates a series of if-then-else decision rules to determine diabetes risk. Each node in the tree represents a test on a health indicator, while branches represent the outcomes of those tests, leading eventually to a diagnosis.
    """)
    
    st.markdown("<h3>About Diabetes</h3>", unsafe_allow_html=True)
    st.markdown("""
    Diabetes is a chronic health condition that affects how your body turns food into energy. If you have diabetes, your body either doesn't make enough insulin or can't use the insulin it makes as well as it should.
    
    There are three main types of diabetes:
    
    - **Type 1 diabetes**: The body does not produce insulin
    - **Type 2 diabetes**: The body doesn't use insulin properly
    - **Gestational diabetes**: Develops during pregnancy
    
    Risk factors for Type 2 diabetes (the most common form) include:
    
    - Being overweight or obese
    - Being 45 years or older
    - Having a family history of diabetes
    - Being physically inactive
    - Having high blood pressure or high cholesterol
    
    Early detection and management of diabetes can prevent serious health complications.
    """)
    
    st.markdown("<h3>Technical Implementation</h3>", unsafe_allow_html=True)
    st.markdown("""
    This project is implemented using:
    
    - **Python**: The core programming language
    - **scikit-learn**: For the decision tree algorithm
    - **pandas**: For data manipulation
    - **Streamlit**: For the web interface
    - **Matplotlib & Seaborn**: For data visualization
    
    The decision tree algorithm uses a recursive binary splitting approach to create a tree where each node represents a feature (health indicator) and each leaf represents a diagnosis.
    """)
    
    st.markdown("<h3>Future Enhancements</h3>", unsafe_allow_html=True)
    st.markdown("""
    Potential improvements to this system could include:
    
    - Integration with electronic health records
    - Addition of more advanced machine learning models (Random Forests, Gradient Boosting)
    - Longitudinal tracking of patient risk factors
    - Mobile application development
    - Incorporation of more detailed medical history and laboratory results
    """)
    
    st.markdown("""
    <div class='footer'>
    <p>Developed as an educational project to demonstrate decision trees in medical diagnosis</p>
    <p>© 2025 - Decision Tree for Medical Diagnosis Project</p>
    </div>
    """, unsafe_allow_html=True)

# Run the app using: streamlit run app.py