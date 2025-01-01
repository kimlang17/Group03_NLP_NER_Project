import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Path to the saved model and tokenizer
output_dir = r"C:\CADT\Data science\Year 4\NLPner_model"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = AutoModelForTokenClassification.from_pretrained(output_dir)

# Define your id2label mapping (adjust based on your model's labels)
id2label = {0: "O", 1: "B", 2: "I"}  # Adjust to match your model's label IDs

# Function to predict NER labels
def predict_ner(text, model, tokenizer, id2label):
    """
    Predicts NER labels for the input text using the model and tokenizer.
    """
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted labels
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()

    # Convert token IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

    # Map predictions to labels
    labels = [id2label[pred] for pred in predictions]

    # Filter out special tokens
    tokens_and_labels = [
        (token, label) for token, label in zip(tokens, labels) if token not in tokenizer.all_special_tokens
    ]

    # Combine subwords and adjust labels
    combined_predictions = []
    current_word = ""
    current_label = "O"

    for token, label in tokens_and_labels:
        if token.startswith("##"):
            # Append subword to the current word
            current_word += token[2:]
        else:
            # Add the previous word to the combined predictions
            if current_word:
                combined_predictions.append((current_word, current_label))
            
            # Start a new word
            current_word = token
            current_label = label

        # Adjust labels based on rules
        if current_label == "I" and (not combined_predictions or combined_predictions[-1][1] == "O"):
            # Correct label to "B" if it's incorrectly labeled as "I"
            current_label = "B"
        elif current_label == "B" and combined_predictions and combined_predictions[-1][1] == "O":
            # Ensure "B" correctly starts a new entity
            current_label = "B"

    # Add the last word to the predictions
    if current_word:
        combined_predictions.append((current_word, current_label))

    return combined_predictions


# Streamlit Interface
def run_streamlit_interface():
    # Set page configuration
    st.set_page_config(page_title="NER Prediction", layout="wide")
    
    # Add a custom header and subtitle
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Named Entity Recognition (NER)</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Predict entity labels from a given sentence</h3>", unsafe_allow_html=True)

    # Styled input box for the sentence
    text = st.text_area(
        "Enter a Sentence for NER Prediction:",
        placeholder="Type your sentence here...",
        height=150,
        max_chars=500,
        help="Enter a sentence, and the model will predict the named entities in it."
    )

    # Apply some styling to buttons
    button_style = """
    <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 30px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)

    # Create a custom button to trigger the prediction
    if st.button("Predict NER"):
        if text.strip():  # Check if the text is not empty
            with st.spinner('Predicting...'):
                # Get predictions
                predictions = predict_ner(text, model, tokenizer, id2label)

                # Display the predictions in a nicely formatted list
                st.subheader("Prediction Results:")
                st.markdown("<h4 style='text-align: center;'>Entities Detected:</h4>", unsafe_allow_html=True)
                
                # Using a simple list format for displaying predictions
                for token, label in predictions:
                    st.markdown(f"<b>{token}</b>: {label}", unsafe_allow_html=True)
        else:
            st.warning("Please enter a sentence to predict.")

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_interface()
