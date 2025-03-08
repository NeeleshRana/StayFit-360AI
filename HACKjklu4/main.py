import streamlit as st
import google.generativeai as genai
import subprocess  # To run external Python scripts
import time
import os
import platform
import csv

# Configure API Key
genai.configure(api_key="AIzaSyBYjEn0XXDLGwRabuREZTLBe346cCOBl8I")  # Replace with your actual API key

def calculate_bmi(weight, height, units="metric"):
    if units == "metric":
        return weight / (height ** 2)
    elif units == "imperial":
        return 703 * weight / (height ** 2)
    else:
        raise ValueError("Invalid unit system. Use 'metric' or 'imperial'.")

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 25:
        return "normal weight"
    elif 25 <= bmi < 30:
        return "overweight"
    else:
        return "obese"

def generate_plan(weight, height, units, preferences, age, gender, activity_level, medical_conditions):
    bmi = calculate_bmi(weight, height, units)
    bmi_category = get_bmi_category(bmi)

    prompt = f"""
   Generate a structured 7-day meal and exercise plan for an individual with a BMI of {bmi:.2f} ({bmi_category}).
    
    - Dietary preferences: {preferences}.
    - Age: {age}, Gender: {gender}, Activity Level: {activity_level}.
    - Medical Conditions: {medical_conditions}.
    
    Include:
    1. Three meals and two snacks per day with approximate calorie counts.
    2. An exercise routine per day (Push-ups, Squats, Bicep Curls, Tricep Pushdowns).
    Format:
    
    **Day 1**  
    - **Meals:** [Breakfast, Snack, Lunch, Snack, Dinner]  
    - **Exercise:** [Push-ups, Squats, Bicep Curls, Tricep Pushdowns]  
    """


    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt)
        return f"### BMI: {bmi:.2f} ({bmi_category})\n\n" + response.text
    except Exception as e:
        return f"Error generating plan: {e}"

# Streamlit UI
st.title("Diet & Exercise Plan Generator 🥗💪")
st.write("Enter your details below to generate a personalized **7-day meal and exercise plan**.")

# User Inputs
units = st.radio("Select Units:", ["Metric (kg, m)", "Imperial (lbs, inches)"])
weight = st.number_input("Weight:", min_value=30.0, max_value=300.0, step=0.1)
height = st.number_input("Height:", min_value=1.0, max_value=2.5, step=0.01)
preferences = st.text_input("Dietary Preferences (e.g., Vegetarian, Keto, Gluten-Free)")
age = st.number_input("Age:", min_value=10, max_value=100, step=1)
gender = st.selectbox("Gender:", ["Male", "Female", "Other"])
activity_level = st.selectbox("Activity Level:", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active"])
medical_conditions = st.text_area("Medical Conditions (if any):")

if st.button("Generate Plan"):
    unit_system = "metric" if "Metric" in units else "imperial"
    plan = generate_plan(weight, height, unit_system, preferences, age, gender, activity_level, medical_conditions)
    st.markdown(plan)

st.subheader("Start Your Exercise Routine 🏋️‍♂️")

def run_exercise(script_name):
    try:
        subprocess.Popen(["python", script_name])
        st.success(f"{script_name} started successfully!")
    except Exception as e:
        st.error(f"Error starting {script_name}: {e}")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Bicep Curl"):
        run_exercise("BicepCurl.py")

    if st.button("Start Squats"):
        run_exercise("Squarts.py")

with col2:
    if st.button("Start Push-ups"):
        run_exercise("PushUp.py")

    if st.button("Start Tricep Pushdown"):
        run_exercise("Triceps.py")

# Additional functionality if needed
st.subheader("Additional Functions")
import Graph  # Ensure Graph.py is in the same directory
if st.button("Generate Graphs"):
    Graph.show_graphs()
