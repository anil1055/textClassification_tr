import streamlit as st

st.set_page_config(page_title="Turkish Text Classification - via AG", page_icon='📖')
st.header("📖Spam Mail Classification - TR")

with st.sidebar:
    hf_key = st.text_input("HuggingFace Access Key", key="hf_key", type="password")

MODEL_SPAM = {
    "albert": "anilguven/albert_tr_turkish_spam_email",  
    "distilbert": "anilguven/distilbert_tr_turkish_spam_email",
    "bert": "anilguven/bert_tr_turkish_spam_email",
    "electra": "anilguven/electra_tr_turkish_spam_email",
}

MODEL_SPAMS = ["albert","distilbert","bert","electra"]

# Use a pipeline as a high-level helper
from transformers import pipeline
# Create a mapping from formatted model names to their original identifiers
def format_model_name(model_key):
    name_parts = model_key
    formatted_name = ''.join(name_parts)  # Join them into a single string with title case
    return formatted_name

formatted_names_to_identifiers = {
    format_model_name(key): key for key in MODEL_SPAM.keys()
}

# Debug to ensure names are formatted correctly
#st.write("Formatted Model Names to Identifiers:", formatted_names_to_identifiers

with st.expander("About this app"):
    st.write(f"""
    1-Choose your model for hotel review analysis (negative or positive).\n
    2-Enter your sample text.\n
    3-And model predict your text's result.
    """)

model_name: str = st.selectbox("Model", options=MODEL_SPAMS)
selected_model = MODEL_SPAM[model_name]

if not hf_key:
    st.info("Please add your HuggingFace Access Key to continue.")
    st.stop()

access_token = hf_key
pipe = pipeline("text-classification", model=selected_model, token=access_token)

#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained(selected_model)
#pipe = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=selected_model)

# Display the selected model using the formatted name
model_display_name = selected_model  # Already formatted
st.write(f"Model being used: `{model_display_name}`")


comment = st.text_input("Enter your mail text for analysis")#User input

st.text('')
if st.button("Submit for Analysis"):
    if not hf_key:
        st.info("Please add your HuggingFace Access Key to continue.")
        st.stop()
    else:
        result = pipe(comment)[0]
        label=''
        if result["label"] == "LABEL_0": label = "Ham mail"
        else: label = "Spam mail"
        st.text(label + " with " + str(result["score"]) + " accuracy")


