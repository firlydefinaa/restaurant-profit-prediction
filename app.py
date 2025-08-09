import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model & encoder
model = joblib.load('model/best_model.pkl')
le_restaurantid = joblib.load('model/le_restaurantid.pkl')
le_menucategory = joblib.load('model/le_menucategory.pkl')
le_menuitem = joblib.load('model/le_menuitem.pkl')
le_ingredients = joblib.load('model/le_ingredients.pkl')
le_target = joblib.load('model/le_target.pkl')

# Load dataset untuk dropdown chaining
df = pd.read_csv('data/restaurant_profitability.csv', sep=';')

# Emoji berdasarkan kategori menu
emoji_map = {
    "Desserts": "🍰",
    "Beverages": "🥤",
    "Main Course": "🍛",
    "Appetizers": "🥗",
    "Fast Food": "🍔",
    "Seafood": "🦞",
    "Pizza": "🍕",
    "Pasta": "🍝",
    "Salads": "🥬"
}

# CSS custom
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #74ebd5, #acb6e5);
        font-family: "Poppins", sans-serif;
    }
    .main-box {
        background-color: white;
        padding: 40px 30px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        max-width: 550px;
        margin: auto;
        text-align: center;
    }
    h2 {
        color: #2980b9;
        font-weight: 600;
        margin-bottom: 30px;
        font-size: 24px;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
    }
    label {
        font-weight: 500;
        color: #34495e;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #2980b9, #6dd5fa);
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #6dd5fa, #2980b9);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 20px;
        font-size: 13px;
        color: #7f8c8d;
    }
    .result-box {
        margin-top: 20px;
        padding: 20px;
        border-radius: 10px;
        font-weight: bold;
        font-size: 22px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        opacity: 0;
        animation: fadeIn 1s forwards;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .high {
        background-color: #eafaf1;
        border: 1px solid #b2f7c1;
        color: #27ae60;
    }
    .medium {
        background-color: #fff9e6;
        border: 1px solid #ffe58f;
        color: #d4a017;
    }
    .low {
        background-color: #fff0f0;
        border: 1px solid #ffb3b3;
        color: #d63031;
    }
    </style>
""", unsafe_allow_html=True)

# Judul + ikon kaca
st.markdown("<h2>🔍 Prediksi Profit Menu Restaurant</h2>", unsafe_allow_html=True)

# Dropdown chaining
restaurant_id = st.selectbox("🏢 Restaurant ID", ["Pilih Restaurant ID"] + sorted(df['RestaurantID'].unique()))

menu_category = None
menu_item = None
ingredients = None
price = None
prediction = None

if restaurant_id != "Pilih Restaurant ID":
    categories = df[df['RestaurantID'] == restaurant_id]['MenuCategory'].unique()
    categories_with_emoji = [f"{emoji_map.get(cat, '🍽️')} {cat}" for cat in sorted(categories)]
    selected_cat_display = st.selectbox("📂 Menu Category", ["Pilih Menu Category"] + categories_with_emoji)

    if selected_cat_display != "Pilih Menu Category":
        menu_category = selected_cat_display.split(" ", 1)[1]
        items = df[(df['RestaurantID'] == restaurant_id) & 
                   (df['MenuCategory'] == menu_category)]['MenuItem'].unique()
        menu_item = st.selectbox("🍴 Menu Item", ["Pilih Menu Item"] + sorted(items))
        
        if menu_item != "Pilih Menu Item":
            match_rows = df[(df['RestaurantID'] == restaurant_id) &
                            (df['MenuCategory'] == menu_category) &
                            (df['MenuItem'] == menu_item)]
            if not match_rows.empty:
                row = match_rows.iloc[0]
                ingredients = row['Ingredients']
                price = row['Price']
                st.text_input("🧾 Ingredients", ingredients, disabled=True)
                st.text_input("💰 Price", price, disabled=True)

# Tombol Prediksi
if st.button("🔍 Prediksi"):
    if restaurant_id not in ["", "Pilih Restaurant ID"] and \
       menu_category not in ["", "Pilih Menu Category", None] and \
       menu_item not in ["", "Pilih Menu Item", None] and \
       price is not None:
        f0 = le_restaurantid.transform([restaurant_id])[0]
        f1 = le_menucategory.transform([menu_category])[0]
        f2 = le_menuitem.transform([menu_item])[0]
        f3 = le_ingredients.transform([ingredients])[0]
        f4 = float(price)

        features = np.array([[f0, f1, f2, f3, f4]])
        pred_label = model.predict(features)[0]
        prediction = le_target.inverse_transform([pred_label])[0]

        css_class = "low"
        if prediction.lower() == "high":
            css_class = "high"
        elif prediction.lower() == "medium":
            css_class = "medium"

        st.markdown(f"<div class='result-box {css_class}'>✅ {prediction}</div>", unsafe_allow_html=True)
    else:
        st.error("⚠️ Harap lengkapi semua pilihan sebelum memprediksi.")

# Footer
st.markdown("<div class='footer'>© 2025 Aplikasi Prediksi Profit - UAS MPML</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
