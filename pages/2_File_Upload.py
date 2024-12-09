import streamlit as st
import pandas as pd

st.set_page_config(page_title="Turkish Text Classification Tasks - via AG", page_icon='📖')
st.header("📖Text Classification for Your File - TR")

with st.sidebar:
    hf_key = st.text_input("HuggingFace Access Key", key="hf_key", type="password")

MODEL_NEWS = {
    "albert": "anilguven/albert_tr_turkish_news",  
    "distilbert": "anilguven/distilbert_tr_turkish_news",
    "bert": "anilguven/bert_tr_turkish_news",
    "electra": "anilguven/electra_tr_turkish_news",
}

MODEL_SPAM = {
    "albert": "anilguven/albert_tr_turkish_spam_email", 
    "distilbert": "anilguven/distilbert_tr_turkish_spam_email",
    "bert": "anilguven/bert_tr_turkish_spam_email",
    "electra": "anilguven/electra_turkish_spam_email",
}

MODELS = ["albert","distilbert","bert","electra"]
MODEL_TASK = ["News Classfication task","Spam Mail Classification Task"]

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
    1-Upload your file as txt or csv file. Each file contains one sample in the each row.\n
    2-Choose your task (news or spam email classification)
    3-Choose your model according to your task analysis.\n
    4-And model predict your text files. \n
    5-Download your test results.
    """)

st.text('')

uploaded_file = st.file_uploader(
    "Upload a csv or txt file",
    type=["csv", "txt"],
    help="Scanned documents are not supported yet!",
)

if not uploaded_file or not hf_key:
    st.stop()


@st.cache_data
def convert_df(df):
    return df.to_csv().encode("utf-8")

datas = [] 
try:
    if uploaded_file.name.lower().endswith(".csv"):
        text = uploaded_file.read().decode("utf-8", errors="replace")
        datas = text.split("\n")
        with st.expander("Show Datas"):
            st.text(datas)
    elif uploaded_file.name.lower().endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="replace")
        datas = text.split("\n")
        with st.expander("Show Datas"):
            st.text(datas)
    else:
        raise NotImplementedError(f"File type {uploaded_file.name.split('.')[-1]} not supported")
except Exception as e:
    st.error("Error reading file. Make sure the file is not corrupted or encrypted")
    st.stop()

task_name: str = st.selectbox("Task", options=MODEL_TASK)
model_select = ''
if task_name == "News Classfication task": model_select = MODEL_NEWS
else: model_select = MODEL_SPAM

model_name: str = st.selectbox("Model", options=MODELS)
selected_model = model_select[model_name]

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

results=[]
txt = ''
labels=[]
accuracies=[]
values=[]
if st.button("Submit for File Analysis"):
    if not hf_key:
        st.info("Please add your HuggingFace Access Key to continue.")
        st.stop()
    else:       
        label=''
        if task_name == "News Classfication task":
            for data in datas:
                result = pipe(data)[0]
                if result["label"] == "LABEL_0": label = "Economy"
                elif result["label"] == "LABEL_1": label = "Magazine"
                elif result["label"] == "LABEL_2": label = "Health"
                elif result["label"] == "LABEL_3": label = "Politics"
                elif result["label"] == "LABEL_4": label = "Sports"
                elif result["label"] == "LABEL_5": label = "Technology"
                else: label = "Events"
                results.append(data[:-1] + ", " + label + ", " + str(result["score"]*100) + "\n")
                labels.append(label)
                accuracies.append(str(result["score"]*100))
                values.append(data[:-1])
                txt += data[:-1] + ", " + label + ", " + str(result["score"]*100) + "\n"
        else:
            for data in datas:
                result = pipe(data)[0]
                if result["label"] == "LABEL_0": label = "Ham mail"
                else: label = "Spam mail"
                results.append(data[:-1] + ", " + label + ", " + str(result["score"]*100) + "\n")
                labels.append(label)
                accuracies.append(str(result["score"]*100))
                values.append(data[:-1])
                txt += data[:-1] + ", " + label + ", " + str(result["score"]*100) + "\n"
        
        st.text("All files evaluated. You'll download result file.")
        if uploaded_file.name.lower().endswith(".txt"):
            with st.expander("Show Results"):
                st.write(results)
            st.download_button('Download Result File', txt, uploaded_file.name.lower()[:-4] + "_results.txt")

        elif uploaded_file.name.lower().endswith(".csv"):
            dataframe = pd.DataFrame({ "text": values,"label": labels,"accuracy": accuracies})
            with st.expander("Show Results"):
                st.write(dataframe)
            csv = convert_df(dataframe)
            st.download_button(label="Download as CSV",data=csv,file_name=uploaded_file.name.lower()[:-4] + "_results.csv",mime="text/csv")
        else:
            raise NotImplementedError(f"File type not supported")



