import streamlit as st
import pandas as pd
import pickle
import numpy as np

with open('item_similarity.pkl', 'rb') as file:
    similarity_df = pickle.load(file)

with open('name_to_code.pkl', 'rb') as f:
    name_to_code = pickle.load(f)

with open('code_to_name.pkl', 'rb') as f:
    code_to_name = pickle.load(f)

with open('rfm_kmeans_model.pkl','rb')as f:
    kmeans_model = pickle.load(f)

with open('product_list.pkl', 'rb') as f:
    product_names = pickle.load(f)

cluster_labels = {
    0: "High-Value",
    1: "Regular Shopper",
    2: "Occasional Shopper",
    3: "At-Risk"
}


def get_top_5_similar(product_code):
    product_codes = list(name_to_code.values())
    if product_code not in product_codes:
        return []

    index = product_codes.index(product_code)
    similarity_scores = similarity_df[index]
    similar_indices = similarity_scores.argsort()[::-1][1:6]  
    recommended_codes = [product_codes[i] for i in similar_indices]

    code_to_name = {v: k for k, v in name_to_code.items()}
    return [code_to_name.get(code, f"Product {code}") for code in recommended_codes]


option = st.sidebar.radio("Home", ["Recommendations", "clustering"])

if option == "Recommendations":
    st.title("🛍️ Product Recommender")
    st.write("Enter a product name")

    product_input = st.text_input("Product Name")

    if st.button("Recommend"):
        normalized_input = product_input.strip().upper()
        normalized_names = [name.strip().upper() for name in product_names]

        if normalized_input in normalized_names:
            idx = normalized_names.index(normalized_input)
            original_name = product_names[idx]
            recommended_products = get_top_5_similar(name_to_code[original_name])

            st.subheader("Recommended Products:")
            for prod in recommended_products:
                st.markdown(f"- {prod}")
        else:
            st.warning("Selected product not found.")

elif option == "clustering":
    st.title("🧠 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=0.01)

    if st.button("Predict Segment"):
        try:
            input_data = np.array([[recency, frequency, monetary]])
            cluster = kmeans_model.predict(input_data)[0]
            labels = cluster_labels.get(cluster, f"Cluster {cluster}")
            st.success(f"This customer belongs to: **{labels}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
