import streamlit as st
import pickle
from sklearn.pipeline import Pipeline

# Load the trained model
with open("rent_pkl", "rb") as file:
    rent_model = pickle.load(file)

# Load the pre-defined pipeline
with open("pipeline_pkl", "rb") as file:
    pipeline = pickle.load(file)

# Define label mappings for categorical features
type_mapping = {'BHK2': 1, 'BHK3': 2, 'BHK1': 0, 'RK1': 5, 'BHK4': 3, 'BHK4PLUS': 4}
building_type_mapping = {'AP': 0, 'IH': 3, 'IF': 2, 'GC': 1}
parking_mapping = {'BOTH': 0, 'TWO_WHEELER': 3, 'NONE': 2, 'FOUR_WHEELER': 1}

def predict_rent(bathroom, property_size, balconies, lift, total_floor, swimming_pool, gym, type, floor, building_type,
                 parking):
    # Map binary categorical features to 1 or 0
    lift_encoded = 1 if lift.lower() == "yes" else 0
    swimming_pool_encoded = 1 if swimming_pool.lower() == "yes" else 0
    gym_encoded = 1 if gym.lower() == "yes" else 0

    # Encode categorical features
    type_encoded = type_mapping.get(type, -1)  # Use -1 for unseen labels
    building_type_encoded = building_type_mapping.get(building_type, -1)  # Use -1 for unseen labels
    parking_encoded = parking_mapping.get(parking, -1)  # Use -1 for unseen labels

    # Check for unseen labels
    if type_encoded == -1 or building_type_encoded == -1 or parking_encoded == -1:
        st.error("Unseen label for categorical feature. Please provide a valid value.")
        return None

    # Transform input features using the pre-defined pipeline
    input_features = [[bathroom, property_size, balconies, lift_encoded, total_floor, swimming_pool_encoded, gym_encoded,
                       type_encoded, floor, building_type_encoded, parking_encoded]]
    input_features_transformed = pipeline.transform(input_features)

    # Make prediction
    predicted_rent = rent_model.predict(input_features_transformed)

    return predicted_rent[0]

# Streamlit app UI
def main():
    st.title("Rent Prediction App")
    st.write("Enter the details below to predict the rent:")

    bathroom = st.text_input("Number of Bathrooms", "2")
    property_size = st.text_input("Property Size (sq. ft.)", "1000")
    balconies = st.text_input("Number of Balconies", "1")
    total_floor = st.text_input("Total Floors", "10")

    lift = st.radio("Lift", ("Yes", "No"))
    swimming_pool = st.radio("Swimming Pool", ("Yes", "No"))
    gym = st.radio("Gym", ("Yes", "No"))

    type = st.selectbox("Type", ['BHK2', 'BHK3', 'BHK1', 'RK1', 'BHK4', 'BHK4PLUS'])
    building_type = st.selectbox("Building Type", ['AP', 'IH', 'IF', 'GC'])
    parking = st.selectbox("Parking", ['BOTH', 'TWO_WHEELER', 'NONE', 'FOUR_WHEELER'])

    floor = st.text_input("Floor", "5")

    if st.button("Predict"):
        predicted_rent = predict_rent(int(bathroom), int(property_size), int(balconies), lift, int(total_floor), swimming_pool, gym, type,
                                      int(floor), building_type, parking)
        if predicted_rent is not None:
            st.success(f"The predicted rent is: {predicted_rent} INR")

if __name__ == "__main__":
    main()
