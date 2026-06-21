import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import json

# --- 1. CONFIGURATION & DATA LOADING ---

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'plant_identifier_resnet50.keras',
        custom_objects={'preprocess_input': tf.keras.applications.resnet50.preprocess_input}
    )

@st.cache_data
def load_data():
    return pd.read_csv('tips.csv')

idx_to_clean_label = {
    0: 'african_violet',
    1: 'aloe_vera',
    2: 'anthurium',
    3: 'areca_palm',
    4: 'asparagus_fern',
    5: 'begonia',
    6: 'bird_of_paradise',
    7: 'birds_nest_fern',
    8: 'boston_fern',
    9: 'calathea',
    10: 'cast_iron_plant',
    11: 'chinese_evergreen',
    12: 'chinese_money_plant',
    13: 'christmas_cactus',
    14: 'chrysanthemum',
    15: 'ctenanthe',
    16: 'daffodils',
    17: 'dracaena',
    18: 'dumb_cane',
    19: 'elephant_ear',
    20: 'english_ivy',
    21: 'hyacinth',
    22: 'iron_cross_begonia',
    23: 'jade_plant',
    24: 'kalanchoe',
    25: 'lilium',
    26: 'lily_of_the_valley',
    27: 'money_tree',
    28: 'monstera_delicosa',
    29: 'orchid',
    30: 'parlor_palm',
    31: 'peace_lily',
    32: 'poinsettia',
    33: 'polka_dot_plant',
    34: 'ponytail_palm',
    35: 'pothos',
    36: 'prayer_plant',
    37: 'rattlesnake_plant',
    38: 'rubber_plant',
    39: 'sago_palm',
    40: 'schefflera',
    41: 'snake_plant',
    42: 'tradescantia',
    43: 'tulip',
    44: 'venus_flytrap',
    45: 'yucca',
    46: 'zz_plant'
}
model = load_model()
tips_df = load_data()

# --- 2. FRONT-END UI ---

st.title("Houseplant Advisor")
st.write("Upload a photo of your houseplant to identify it and get tips on how to care for it.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Your Plant', use_container_width=True)
    
    # --- 3. PREPROCESSING & PREDICTION ---
    
    # Ensure image is in RGB format (prevents crashes from PNGs with transparent backgrounds)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # --- 4. DISPLAY RESULTS & TIPS ---
    
    if confidence > 30:
        clean_label = idx_to_clean_label.get(predicted_index, "Unknown")
        
        # Format the label to look nice (e.g., "snake_plant" -> "Snake plant")
        formatted_label = clean_label.replace('_', ' ').capitalize()
        st.success(f"I am {confidence:.2f}% sure this is a {formatted_label}")
        
        tips = tips_df.loc[tips_df['clean_label'] == clean_label]
        
        if not tips.empty:
            row = tips.iloc[0]
            
            for col in ['common_name', 'water', 'light', 'tips']:
                # Defensive check to make sure the column actually exists in your CSV
                if col in row: 
                    expander_title = col.replace('_', ' ').capitalize()
                    with st.expander(f"{expander_title}", expanded=True):
                        st.write(row[col])
        else:
            st.error("No care tips found for this plant in the database.")
            
    else:
        st.warning("I'm not confident enough to identify this plant.")