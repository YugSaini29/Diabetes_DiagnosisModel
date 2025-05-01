# Diabetes_DiagnosisModel
How to Run the Diabetes Decision Tree Web Application
Follow these steps to get the web interface up and running:

1. Clone or Download the Repository
Clone or download the repository containing the decision tree web application. If you have Git installed, you can run:
git clone <repository-url>
Alternatively, download the ZIP file and extract it to your desired location.

2. Install Required Libraries
Before running the app, you’ll need to install the necessary Python packages. Open a terminal/command prompt and navigate to the folder where the Python scripts are located.

Run the following command to install all required dependencies:
pip install streamlit pydotplus ipython
Additional Dependencies: If your script uses any other libraries, make sure they are also installed. You can check the imports at the top of the decision_tree_web_interface.py file and install any missing packages using pip install <package-name>.

3. Run the Web Application
Once all libraries are installed, navigate to the folder containing the decision_tree_web_interface.py script.

cd path/to/your/script 
After that, run the Streamlit app using:
streamlit run decision_tree_web_interface.py
just paste this into your vscode terminal, it will open the steamlit app
This will start a local server. After a few moments, you should see the following output in the terminal:

You can now view your Streamlit app in your browser.
Local URL:  http://localhost:8501
Network URL: http://<your-ip>:8501
4. Open the Application in Your Browser
Open a web browser and navigate to:
http://localhost:8501
You should see the diabetes decision tree web application, where you can input patient data, get a diagnosis, and visualize the decision-making process.

5. Troubleshooting
Missing Packages: If you encounter an error about missing packages, install them using pip install <missing-package>.

Decision Tree Rendering Issues: If the decision tree visualization doesn’t work, ensure that Graphviz is installed on your machine. Follow the installation steps for Graphviz if necessary.
