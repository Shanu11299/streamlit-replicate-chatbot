import streamlit as st
import replicate

# Set up Replicate API key
REPLICATE_API_TOKEN = "r8_B6xXRy9R43ErvRfCko033QtoKeHetnM3L1q9i"  # Replace with your actual key
replicate.client.set_token(REPLICATE_API_TOKEN)

# Streamlit UI
st.title("🤖 AI Chatbot Using Replicate")

# Store messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
user_input = st.chat_input("Ask me anything!")

if user_input:
    # Add user input to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call Replicate API (Example: Using Mistral 7B Chat Model)
    response = replicate.run(
        "mistralai/mistral-7b-instruct-v0.1",
        input={"prompt": user_input}
    )
    
    # Get model response
    bot_reply = response[0]  # Extract text from API response

    # Display bot response
    with st.chat_message("assistant"):
        st.write(bot_reply)

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
