'''import pickle
import streamlit as st
import pandas as pd
from PIL import Image

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

def main():
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')

    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.sidebar.image(image2, use_column_width=True)
    st.sidebar.title("Customer Churn Prediction")
    st.sidebar.info('This app is created to predict Customer Churn')

    st.image(image, use_column_width=False)
    st.title("Predicting Customer Churn")

    add_selectbox = st.sidebar.radio("How would you like to predict?", ("Online", "Batch"))

    if add_selectbox == 'Online':
        st.sidebar.subheader("Enter Customer Details")
        gender = st.selectbox('Gender:', ['Male', 'Female'])
        seniorcitizen= st.selectbox('Customer is a senior citizen:', ['No', 'Yes'])
        partner= st.selectbox('Customer has a partner:', ['No', 'Yes'])
        dependents = st.selectbox('Customer has dependents:', ['No', 'Yes'])
        phoneservice = st.selectbox('Customer has phone service:', ['No', 'Yes'])
        multiplelines = st.selectbox('Customer has multiple lines:', ['No phone service', 'No', 'Yes'])
        internetservice= st.selectbox('Customer has internet service:', ['DSL', 'Fiber optic', 'No'])
        onlinesecurity= st.selectbox('Customer has online security:', ['No internet service', 'No', 'Yes'])
        onlinebackup = st.selectbox('Customer has online backup:', ['No internet service', 'No', 'Yes'])
        deviceprotection = st.selectbox('Customer has device protection:', ['No internet service', 'No', 'Yes'])
        techsupport = st.selectbox('Customer has tech support:', ['No internet service', 'No', 'Yes'])
        streamingtv = st.selectbox('Customer has streaming TV:', ['No internet service', 'No', 'Yes'])
        streamingmovies = st.selectbox('Customer has streaming movies:', ['No internet service', 'No', 'Yes'])
        contract= st.selectbox('Customer has a contract:', ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox('Customer has paperless billing:', ['No', 'Yes'])
        paymentmethod= st.selectbox('Payment Option:', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
        tenure = st.number_input('Number of months with current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges= st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure*monthlycharges

        input_dict={
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        if st.button("Predict"):
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

    elif add_selectbox == 'Batch':
        st.sidebar.subheader("Upload CSV File for Batch Prediction")
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform(data)
            y_pred = model.predict_proba(X)[:, 1] >= 0.5
            churn = pd.Series(y_pred, name='Churn')
            st.write(churn)

if __name__ == '__main__':
    main()


'''


import pickle
import streamlit as st
import pandas as pd
from PIL import Image

# Load model and data
model_file = 'model_C=1.0.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# Define function for the main application
def main():
    # Load images
    image = Image.open('images/icone.png')
    image2 = Image.open('images/image.png')

    # Set page configuration
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar
    st.sidebar.image(image2, use_column_width=True)
    st.sidebar.title("Customer Churn Prediction")
    st.sidebar.info('This app is created to predict Customer Churn')

    # Main content
    st.image(image, use_column_width=False)
    st.title("Predicting Customer Churn")

    # Prediction method selection
    add_selectbox = st.sidebar.radio("How would you like to predict?", ("Online", "Batch"))

    # Online prediction
    if add_selectbox == 'Online':
        st.sidebar.subheader("Enter Customer Details")
        # User input fields
        gender = st.selectbox('Gender:', ['Male', 'Female'])
        seniorcitizen= st.selectbox('Customer is a senior citizen:', ['No', 'Yes'])
        partner= st.selectbox('Customer has a partner:', ['No', 'Yes'])
        dependents = st.selectbox('Customer has dependents:', ['No', 'Yes'])
        phoneservice = st.selectbox('Customer has phone service:', ['No', 'Yes'])
        multiplelines = st.selectbox('Customer has multiple lines:', ['No phone service', 'No', 'Yes'])
        internetservice= st.selectbox('Customer has internet service:', ['DSL', 'Fiber optic', 'No'])
        onlinesecurity= st.selectbox('Customer has online security:', ['No internet service', 'No', 'Yes'])
        onlinebackup = st.selectbox('Customer has online backup:', ['No internet service', 'No', 'Yes'])
        deviceprotection = st.selectbox('Customer has device protection:', ['No internet service', 'No', 'Yes'])
        techsupport = st.selectbox('Customer has tech support:', ['No internet service', 'No', 'Yes'])
        streamingtv = st.selectbox('Customer has streaming TV:', ['No internet service', 'No', 'Yes'])
        streamingmovies = st.selectbox('Customer has streaming movies:', ['No internet service', 'No', 'Yes'])
        contract= st.selectbox('Customer has a contract:', ['Month-to-month', 'One year', 'Two year'])
        paperlessbilling = st.selectbox('Customer has paperless billing:', ['No', 'Yes'])
        paymentmethod= st.selectbox('Payment Option:', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
        tenure = st.number_input('Number of months with current telco provider:', min_value=0, max_value=240, value=0)
        monthlycharges= st.number_input('Monthly charges:', min_value=0, max_value=240, value=0)
        totalcharges = tenure*monthlycharges

        # Create input dictionary
        input_dict={
            "gender": gender,
            "seniorcitizen": seniorcitizen,
            "partner": partner,
            "dependents": dependents,
            "phoneservice": phoneservice,
            "multiplelines": multiplelines,
            "internetservice": internetservice,
            "onlinesecurity": onlinesecurity,
            "onlinebackup": onlinebackup,
            "deviceprotection": deviceprotection,
            "techsupport": techsupport,
            "streamingtv": streamingtv,
            "streamingmovies": streamingmovies,
            "contract": contract,
            "paperlessbilling": paperlessbilling,
            "paymentmethod": paymentmethod,
            "tenure": tenure,
            "monthlycharges": monthlycharges,
            "totalcharges": totalcharges
        }

        # Predict button
        if st.button("Predict", key="predict_button"):
            X = dv.transform([input_dict])
            y_pred = model.predict_proba(X)[0, 1]
            churn = y_pred >= 0.5
            output_prob = float(y_pred)
            output = bool(churn)
            st.success('Churn: {0}, Risk Score: {1}'.format(output, output_prob))

    # Batch prediction
    elif add_selectbox == 'Batch':
        st.sidebar.subheader("Upload CSV File for Batch Prediction")
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            X = dv.transform(data)
            y_pred = model.predict_proba(X)[:, 1] >= 0.5
            churn = pd.Series(y_pred, name='Churn')
            st.write(churn)

if __name__ == '__main__':
    main()
