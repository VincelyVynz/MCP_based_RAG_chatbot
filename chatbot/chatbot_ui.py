import streamlit as st
import asyncio
from chatbot.chat import answer_query, update_employee

st.set_page_config(page_title = "RAG Chatbot with MCP", page_icon= "🤖")
st.title("RAG Chatbot with MCP")

st.sidebar.header("Update Employee Info")
employee_name = st.sidebar.text_input("Employee Name")
field = st.sidebar.text_input("Field to update")
new_value = st.sidebar.text_input("New Value")
update_btn = st.sidebar.button("Update Employee")

if update_btn:
    if employee_name and field and new_value:
        result = asyncio.run(update_employee(employee_name, field, new_value))
        st.sidebar.success(result)
    else:
        st.sidebar.error("Please fill all fields to update.")

#Chat
st.header("Ask anything")
user_query = st.text_input("Type here")
ask_btn = st.button("➡️")

if ask_btn and user_query:
    with st.spinner("Generating response..."):
        answer = asyncio.run(answer_query(user_query))
        st.markdown("**" + answer + "**")