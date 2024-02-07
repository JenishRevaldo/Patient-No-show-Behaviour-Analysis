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

    scheduled_day = st.date_input("Scheduled Day")
    appointment_day = st.date_input("Appointment Day")


    neighbourhood_options = original_df['Neighbourhood'].unique()
    neighbourhood = st.selectbox("Neighbourhood", neighbourhood_options)

    sms_received = st.selectbox("SMS Received", ["No", "Yes"])

    # Validate PatientId against the original dataset
    if st.button("Submit"):
        
        patient_data = original_df[original_df['PatientId'] == patient_id].iloc[0]

        # Create a dictionary with the form data
        form_data = {
            "PatientId": patient_data['PatientId'],
            "Gender": patient_data['Gender'],
            "ScheduledDay": scheduled_day,
            "AppointmentDay": appointment_day,
            "Age": patient_data['Age'],
            "Neighbourhood": neighbourhood,
            "Scholarship": patient_data['Scholarship'],
            "Hipertension": patient_data['Hipertension'],
            "Diabetes": patient_data['Diabetes'],
            "Alcoholism": patient_data['Alcoholism'],
            "Handcap": patient_data['Handcap'],
                "SMS_received": 1 if sms_received == "Yes" else 0,
            }

        # Convert the form data to a DataFrame for further processing

        df = pd.DataFrame([form_data])
        df['Gender'] = df['Gender'].map({'M' : 1, 'F' : 0})
       
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
            prediction = 'This Person may not show up for the appoinment'
        else:
            prediction = 'This Person may show up for the appoinment'
        
        st.markdown(f"**Patient Details:**\n\n"
                        f"- Patient ID: {form_data['PatientId']}\n"
                        f"- Scheduled Day: {form_data['ScheduledDay']}\n"
                        f"- Appointment Day: {form_data['AppointmentDay']}\n"
                        f"- Gender: {form_data['Gender']}\n"
                        f"- Age: {form_data['Age']}\n"
                        f"- Neighbourhood: {form_data['Neighbourhood']}\n"
                        f"- Scholarship: {'Yes' if form_data['Scholarship'] == 1 else 'No'}\n"
                        f"- Hipertension: {'Yes' if form_data['Hipertension'] == 1 else 'No'}\n"
                        f"- Diabetes: {'Yes' if form_data['Diabetes'] == 1 else 'No'}\n"
                        f"- Alcoholism: {'Yes' if form_data['Alcoholism'] == 1 else 'No'}\n"
                        f"- Handicap: {'Yes' if form_data['Handcap'] == 1 else 'No'}\n"
                        f"- SMS Received: {'Yes' if form_data['SMS_received'] == 1 else 'No'}\n\n"
                        f"**Prediction:** {prediction}")

if __name__ == "__main__":
    main()
