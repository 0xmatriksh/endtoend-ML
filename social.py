import streamlit as sl
import pandas as pd
import pickle
import sklearn

pipe  = pickle.load(open('pipe.pkl','rb'))

sl.title('Social Network Ad')

gender = sl.selectbox('Select the Gender',['Male','Female'])

age = sl.number_input('Enter the age : ')

salary = sl.number_input('Enter the estimated salary')

sl.text(sklearn.__version__)

if sl.button('Predict'):
    input_df = pd.DataFrame({'Age':[age],'EstimatedSalary':[salary],
    'Gender':[gender]})
    result = pipe.predict_proba(input_df)

    # in result first one is negative probability
    sl.text(f"Probability :{round(result[0][1]*100)} %")





