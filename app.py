
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Sample Dataset
data = {
    'Study_Hours': [2, 5, 1, 8, 3, 6, 4, 7],
    'Attendance': [60, 80, 50, 90, 65, 85, 70, 95],
    'Previous_Marks': [50, 70, 40, 88, 60, 75, 68, 92],
    'Pass': [0, 1, 0, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance', 'Previous_Marks']]
y = df['Pass']

model = LogisticRegression()
model.fit(X, y)

# Web App UI
st.title("🎓 Student Performance Prediction")

study_hours = st.number_input("Enter Study Hours", 0, 12)
attendance = st.number_input("Enter Attendance (%)", 0, 100)
previous_marks = st.number_input("Enter Previous Marks", 0, 100)

if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance, previous_marks]])
    
    if prediction[0] == 1:
        st.success("✅ The student will PASS")
    else:
        st.error("❌ The student will FAIL")
