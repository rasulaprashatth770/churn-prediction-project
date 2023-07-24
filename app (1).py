import streamlit as st
import pickle
import numpy as np


st.set_page_config(page_title="TELECOM_CHURN", page_icon=":tada:", layout="centered")


# Load the random forest model from the pickle file
with open('random_search_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data from the pickle file
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
    

# Create a function to make predictions
def predict_churn(international_plan, voice_mail_plan, account_length, international_mins,
                  international_calls, days_mins, days_calls, eve_mins,
                  eve_calls, night_calls, night_minutes,total_calls,
                  customer_service_calls):
    # Map "Yes" or "No" to 1 or 0
    international_plan = 1 if international_plan == "Yes" else 0
    voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0

    # Create a NumPy array from the input values
    input_data = np.array([[international_plan, voice_mail_plan, account_length,
                            international_mins, international_calls, days_mins,
                            days_calls, eve_mins, eve_calls,
                            night_calls, night_minutes,total_calls,
                            customer_service_calls]])

    # Make the churn prediction
    prediction = model.predict(input_data)[0]
    churn_probability = model.predict_proba(input_data)[0][1]  # Probability of churn (positive class)
    staying_probability = 1 - churn_probability  # Probability of staying (negative class)
    
    if prediction == 0:
            st.write("The customer is likely to stay." 
                     f" Staying Probability: {staying_probability:.2%}")
    else:
            st.write("The customer is likely to churn. "
                     f"Churn Probability: {churn_probability:.2%}")
        
    #st.write(f"Churn Probability: {churn_probability:.2%}")

# Create the Streamlit app
def main():
    st.title("Churn Prediction App")

    # Create input fields for user input
    international_plan = st.selectbox('International plan', ['No', 'Yes'])
    voice_mail_plan = st.selectbox('Voice mail plan', ['No', 'Yes'])
    account_length = st.number_input('Account length', min_value=1.0, max_value=300.0, step=1.0)
    international_mins = st.number_input('Internation Minutes', min_value=0.0, max_value=100.0, step=1.0)
    international_calls = st.number_input('Internation Calls', min_value=0.0, max_value=3000.0, step=1.0)
    days_mins = st.number_input('Day Minutes', min_value=0.0, max_value=3000.0, step=1.0)
    days_calls = st.number_input('Day Calls', min_value=0.0, max_value=3000.0, step=1.0)
    eve_mins = st.number_input('Evening Minutes', min_value=0.0, max_value=3000.0, step=1.0)
    eve_calls = st.number_input('Evening Calls', min_value=0.0, max_value=3000.0, step=1.0)
    night_calls = st.number_input('Night Calls', min_value=0.0, max_value=3000.0, step=1.0)
    night_minutes = st.number_input('Night Minutes', min_value=0.0, max_value=3000.0, step=1.0)
    total_calls = st.number_input('Total Calls', min_value=0.0, max_value=3000.0, step=1.0)
    customer_service_calls = st.number_input('Customer service calls', min_value=0.0, max_value=30.0, step=1.0)
    

    # When the user clicks the "Predict" button, make the prediction
    if st.button("Predict"):
        prediction = predict_churn(international_plan, voice_mail_plan, account_length,
                                   international_mins, international_calls, days_mins,
                                   days_calls, eve_mins, eve_calls,
                                   night_calls, night_minutes, total_calls,
                                   customer_service_calls)
        st.write(prediction)
        

if __name__ == "__main__":
    main()
