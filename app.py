import streamlit as st
import pandas as pd
import numpy
import pickle


# Load the original dataset
original_df = pd.read_csv('Datasets/Patient no show.csv')
city_stats = pd.read_csv('Datasets/city_stats.csv')
preferred_day = pd.read_csv('Datasets/preferred_day.csv')

# Load your trained model using pickle
with open('MLP1model.pkl', 'rb') as file:
    model = pickle.load(file)

def main():
    st.title("Appointment Booking Form")

    # Dropdown menu for PatientId
    patient_id_options = original_df['PatientId'].unique()
    patient_id = st.selectbox("Patient ID", patient_id_options)

    gender = st.selectbox("Gender", ["Male", "Female"])
    scheduled_day = st.date_input("Scheduled Day")
    appointment_day = st.date_input("Appointment Day")
    age = st.number_input("Age", value=0)

    neighbourhood_options = original_df['Neighbourhood'].unique()
    neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

    scholarship = st.selectbox("Scholarship", ["No", "Yes"])
    hipertension = st.selectbox("Hipertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    alcoholism = st.selectbox("Alcoholism", ["No", "Yes"])
    handicap = st.selectbox("Handicap", ["No", "Yes"])
    sms_received = st.selectbox("SMS Received", ["No", "Yes"])

    # Validate PatientId against the original dataset
    if st.button("Submit"):
        # Create a dictionary with the form data
        form_data = {
            "PatientId": patient_id,
            "Gender": gender,
            "ScheduledDay": scheduled_day,
            "AppointmentDay": appointment_day,
            "Age": age,
            "Neighbourhood": neighbourhood,
            "Scholarship": 1 if scholarship == "Yes" else 0,
            "Hipertension": 1 if hipertension == "Yes" else 0,
            "Diabetes": 1 if diabetes == "Yes" else 0,
            "Alcoholism": 1 if alcoholism == "Yes" else 0,
            "Handcap": 1 if handicap == "Yes" else 0,
            "SMS_received": 1 if sms_received == "Yes" else 0,
        }

        # Convert the form data to a DataFrame for further processing

        df = pd.DataFrame([form_data])
        df['Gender'] = df['Gender'].map({'Male' : 1, 'Female' : 0})
       
        df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
        df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

        df['Date_of_Reg'] = df['ScheduledDay'].dt.date
        df['Date_of_App'] = df['AppointmentDay'].dt.date

        df['Month_of_Reg'] = df['ScheduledDay'].dt.month
        df['Month_of_App'] = df['AppointmentDay'].dt.month

        df['Day_of_Reg'] = df['ScheduledDay'].dt.dayofweek
        df['Day_of_App'] = df['AppointmentDay'].dt.dayofweek

        df['Diff_in_Date'] = (df['Date_of_App'] - df['Date_of_Reg']).dt.days.astype(int)
        df['Diff_in_Month'] = df['Month_of_App'] - df['Month_of_Reg']
        
        df = pd.merge(df, city_stats[['Neighbourhood', 'Popularity_Score']], on='Neighbourhood', how='left')
        df = pd.merge(df, preferred_day, on='PatientId', how='left')
        
        X = df[['Gender','Age','Scholarship','Hipertension','Diabetes','Alcoholism','Handcap',
                'SMS_received','Day_of_Reg','Day_of_App','Diff_in_Date','Diff_in_Month',
                'Popularity_Score','Preferred_Day']]
        
        prediction = model.predict(X)
        
        if prediction == 1:
            prediction = 'The Person may not show up for the appoinment'
        else:
            prediction = 'The Person may show up for the appoinment'
        
        st.write("Prediction :")
        st.write(prediction)

if __name__ == "__main__":
    main()
