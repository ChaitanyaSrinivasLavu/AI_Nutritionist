import streamlit as st
import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpStatus
from joblib import load
import matplotlib.pyplot as plt  # Import matplotlib

# Load the saved model
model = load('random_forest_regressor.joblib')

# Load the food dataset
df = pd.read_csv(r"C:\Users\lavuc\fitness caloriess\fitness calorie\nutrition_updated.csv")  # Replace with your actual file path
df = df[df['calories'] > 0]  # Filter out foods with zero or negative calories

# Prediction function
def predict_calories(Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp):
    input_data = pd.DataFrame([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    prediction = model.predict(input_data)
    return prediction[0]

# New weight calculation function
def calculate_new_weight(current_weight, total_calories):
    # Assuming 7700 calories = 1 kg of body weight change
    weight_change = total_calories / 770
    new_weight = current_weight - weight_change
    return new_weight

# Optimization function for food recommendations
def recommend_foods(predicted_calories, weight_difference_percentage):
    # Calculate adjusted calorie limits
    min_calories = (80 + weight_difference_percentage) * predicted_calories / 100
    max_calories = (100 + weight_difference_percentage) * predicted_calories / 100

    # Prepare data
    df['calories'] = pd.to_numeric(df['calories'], errors='coerce').fillna(0)
    df['protein'] = pd.to_numeric(df['protein'], errors='coerce').fillna(0)

    solutions = []
    excluded_items = set()

    for _ in range(3):  # Generate at least 3 sets of recommendations
        problem = LpProblem("Maximize_Protein", LpMaximize)
        choices = [LpVariable(f"choice_{i}", cat="Binary") for i in range(len(df))]

        # Objective: Maximize protein content
        problem += lpSum(choices[i] * df.iloc[i]['protein'] for i in range(len(df)))

        # Constraints
        total_calories = lpSum(choices[i] * df.iloc[i]['calories'] for i in range(len(df)))
        problem += total_calories >= min_calories  # Minimum calorie limit
        problem += total_calories <= max_calories  # Maximum calorie limit
        problem += lpSum(choices) == 3  # Select exactly 3 items

        for i in excluded_items:
            problem += choices[i] == 0  # Exclude already selected items

        # Solve the problem
        status = problem.solve()
        if LpStatus[status] != "Optimal":
            break

        # Extract selected items
        selected_items = [i for i in range(len(choices)) if choices[i].varValue == 1]
        excluded_items.update(selected_items)
        solutions.append(selected_items)

    if not solutions:
        return None

    # Return solutions as DataFrame
    recommendation_sets = []
    for solution in solutions:
        selected_foods = df.iloc[solution][['name', 'calories', 'protein']]
        recommendation_sets.append(selected_foods)
    return recommendation_sets

# Streamlit App UI
st.title("Calories Burnt Prediction and Food Recommendation")

st.sidebar.header("Input Parameters")
Gender = st.sidebar.slider("Gender (0 = Female, 1 = Male)", min_value=0, max_value=1, step=1)
Age = st.sidebar.number_input("Age", min_value=10, max_value=100, step=1)
Height = st.sidebar.number_input("Height (in cm)", min_value=50.0, max_value=250.0, step=1.0)
Weight = st.sidebar.number_input("Weight (in kg)", min_value=10.0, max_value=300.0, step=1.0)
Duration = st.sidebar.number_input("Duration (in minutes)", min_value=1, max_value=300, step=1)
Heart_Rate = st.sidebar.number_input("Heart Rate (bpm)", min_value=40, max_value=200, step=1)
Body_Temp = st.sidebar.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, step=0.1)
weight_difference_percentage = st.sidebar.slider("Diet Intensity", min_value=-50, max_value=0, step=1)
target_weight = st.sidebar.number_input("Target Weight (kg)", min_value=10.0, max_value=300.0, step=0.1)

if st.sidebar.button("Predict and Recommend"):
    with st.spinner("Processing..."):
        current_weight = Weight
        day = 1
        weight_progress = []  # To track weight changes over days

        # Iterate until target weight is reached
        while current_weight > target_weight:
            st.write(f"**Day {day}:**")
            
            # Predict calories burnt
            predicted_calories = predict_calories(Gender, Age, Height, current_weight, Duration, Heart_Rate, Body_Temp)
            st.success(f"Predicted Calories Burnt: {predicted_calories:.2f}")

            # Get food recommendations
            recommendation_sets = recommend_foods(predicted_calories, weight_difference_percentage)

            if recommendation_sets is None:
                st.error("No feasible food recommendations found.")
                break
            else:
                for idx, recommended_df in enumerate(recommendation_sets):
                    st.write(f"**Recommendation Set {idx+1}:**")
                    st.dataframe(recommended_df)

                    # Calculate and display totals
                    total_calories = recommended_df['calories'].sum()
                    total_protein = recommended_df['protein'].sum()
                    st.write(f"Total Calories: {total_calories:.2f}")
                    st.write(f"Total Protein: {total_protein:.2f}")

                    # Calculate new weight
                    new_weight = calculate_new_weight(current_weight, total_calories)
                    st.success(f"Estimated New Weight (Set {idx+1}): {new_weight:.2f} kg")

                    # Update current weight for next iteration
                    current_weight = new_weight

            # Add current weight to the progress list
            weight_progress.append(current_weight)

            # Increment the day counter
            day += 1

            # Stop if target weight is reached
            if current_weight <= target_weight:
                st.success(f"Target weight of {target_weight:.2f} kg reached!")
                break

        # Plot the graph of Day vs Weight
        st.write("### Weight Progress Over Days")
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, day), weight_progress, marker='o', linestyle='-', color='b')
        plt.title("Weight Progress Over Days")
        plt.xlabel("Day")
        plt.ylabel("Weight (kg)")
        plt.grid(True)
        st.pyplot(plt)
