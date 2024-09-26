import streamlit as st
import requests

# Streamlit application
st.title("RAG Application UI")

# Input box for the user's query
query = st.text_input("Enter your question about the PDF content:")

if st.button("Submit"):
    if query:
        # Make a request to the Flask API
        response = requests.post(
            "http://127.0.0.1:5000/api/ask",  # Ensure this matches your Flask API URL
            json={"query": query},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            answer = response.json().get("answer")
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.error("Something went wrong. Please try again.")
    else:
        st.error("Please enter a question.")
