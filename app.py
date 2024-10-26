# # app.py

# import streamlit as st
# import pandas as pd
# from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
# import sentencepiece as spm

# # Load tokenizer, model, and sentencepiece processor
# # tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/odia-bert")
# # model = AutoModelForMaskedLM.from_pretrained("l3cube-pune/odia-bert")
# # pipe = pipeline("fill-mask", model=model)

# # Load tokenizer, model, and sentencepiece processor
# tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/odia-bert")
# model = AutoModelForMaskedLM.from_pretrained("l3cube-pune/odia-bert")
# pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)  # Explicitly pass tokenizer and model

# sp = spm.SentencePieceProcessor()
# sp.Load("oriya_lm.model")

# # Streamlit app title and description
# st.title("Odia Sentence Processor")
# st.write("A simple web app to process sentences in the Odia language using NLP models.")

# # File upload section
# uploaded_file = st.file_uploader("Upload a JSONL file with 'sentence' key", type=["jsonl"])

# if uploaded_file is not None:
#     # Read JSONL file into DataFrame
#     df = pd.read_json(uploaded_file, lines=True)
#     st.write("Preview of uploaded data:")
#     st.write(df.head())
    
#     # Extract the 'sentence' column
#     df_reqd = df["sentence"]
    
#     # Select sentence to process
#     sentence = st.selectbox("Choose a sentence from the uploaded file to process:", df_reqd)
# else:
#     # Allow user to enter text directly
#     sentence = st.text_input("Or, enter a sentence to process:")

# if sentence:
#     st.write("### Selected Sentence")
#     st.write(sentence)

#     # Tokenize and process the sentence using SentencePiece
#     words = sp.EncodeAsPieces(sentence)
#     all_letters = [list(word[1:]) for word in words]  # Process tokens into letter splits
    
#     # Display results
#     st.write("### Tokenized Words")
#     st.write(words)

#     st.write("### Split Letters")
#     st.write(all_letters)

#     # Example of masked language model prediction
#     st.write("### Masked Prediction")
#     masked_sentence = sentence.replace(" ", " [MASK] ", 1)  # Add a simple mask for demonstration
#     predictions = pipe(masked_sentence)

#     # Show predictions
#     st.write("Top predictions for the masked sentence:")
#     for pred in predictions:
#         st.write(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}")


import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
import sentencepiece as spm

# Load tokenizer, model, and sentencepiece processor
tokenizer = AutoTokenizer.from_pretrained("l3cube-pune/odia-bert")
model = AutoModelForMaskedLM.from_pretrained("l3cube-pune/odia-bert")
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer)  # Explicitly pass both

# Load sentencepiece model
sp = spm.SentencePieceProcessor()
sp.Load("oriya_lm.model")

# Streamlit app title and description
st.title("Odia Sentence Processor")
st.write("A simple web app to process sentences in the Odia language using NLP models.")

# File upload section
uploaded_file = st.file_uploader("Upload a JSONL file with 'sentence' key", type=["jsonl"])

if uploaded_file is not None:
    # Read JSONL file into DataFrame
    df = pd.read_json(uploaded_file, lines=True)
    if "sentence" in df.columns:
        st.write("Preview of uploaded data:")
        st.write(df.head())

        # Extract the 'sentence' column
        df_reqd = df["sentence"]

        # Select sentence to process
        sentence = st.selectbox("Choose a sentence from the uploaded file to process:", df_reqd)
    else:
        st.error("Uploaded file does not contain 'sentence' key.")
        sentence = None
else:
    # Allow user to enter text directly
    sentence = st.text_input("Or, enter a sentence to process:")

if sentence:
    st.write("### Selected Sentence")
    st.write(sentence)

    # Tokenize and process the sentence using SentencePiece
    words = sp.EncodeAsPieces(sentence)
    all_letters = [list(word[1:]) for word in words]  # Process tokens into letter splits

    # Display results
    st.write("### Tokenized Words")
    st.write(words)

    st.write("### Split Letters")
    st.write(all_letters[:20])  # Display only first 20 items to avoid overload

    # Example of masked language model prediction
    st.write("### Masked Prediction")
    masked_sentence = sentence.replace(sentence.split()[0], "[MASK]", 1)
    try:
        predictions = pipe(masked_sentence)
        # Show predictions
        st.write("Top predictions for the masked sentence:")
        for pred in predictions:
            st.write(f"Token: {pred['token_str']}, Score: {pred['score']:.4f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
