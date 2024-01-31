# Medical Appointment No-Show Analysis

This repository contains an analysis of no-show behavior in a dataset consisting of 110,527 medical appointments with 14 associated variables. The primary goal is to understand the factors contributing to patient no-shows and identify patterns that explain this behavior.

## Dataset Overview

- The dataset includes information on 14 variables related to medical appointments.
- The primary outcome of interest is whether a patient shows up or no-shows for their appointment.
- Various factors such as age, gender, appointment month, and SMS reminders are explored.

## Analysis Goals

- **Objective:** Understand the reasons behind patient non-compliance with medical appointments.
- **Approach:** Analyze patterns, identify key variables, and propose potential interventions to improve attendance rates.

## Key Findings

- The data reveals a significant number of patients exhibiting no-show behavior.
- Specific variables are examined for their impact on attendance rates.

## Repository Structure

- `Datasets/`: Contains the dataset used for analysis and the datasets created by me during analysis. *Patient no show* is the file name of raw data.
- `Notebooks/`: Jupyter notebooks with detailed analyses and visualizations.
- `MLP1model.pkl`: Pickle file containing the trained Multilayer Perceptron Model
- `app.py`: Streamlit web application for deployment
- `requirements.txt`: Text file containing the required libraries
- `README.md`: Overview of the repository and analysis goals.

## Web Application

The `app.py` script implements a Streamlit web application for real-time patient show up prediction.


## Getting Started

1. Clone this repository:

   ```bash
   git clone https://github.com/JenishRevaldo/Patient-No-show-Behaviour-Analysis.git
   cd Patient-No-show-Behaviour-Analysis
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Explore the Jupyter notebook (`Notebooks/Patient_No_show_Behaviour_Analysis_1.ipynb`) for insights into data analysis and model development. Also explore other notebooks too.

4. Run the Flask web application:

   ```bash
   streamlit run app.py
   ```

   Visit `http://localhost:8501` in your web browser to access the patient show up prediction application.

## Contributions
Contributions are welcome! If you identify improvements or additional insights, please feel free to open an issue or submit a pull request.

Explore the notebooks in the notebooks/ directory for detailed analyses.
