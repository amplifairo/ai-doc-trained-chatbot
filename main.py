import streamlit as st
import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
# import socket


openai_api_key = None
if openai in st.secrets and api_key in st.secrets.openai:
  openai_api_key = st.secrets.openai.api_key

if not openai_api_key:
  import constants
  openai_api_key = constants.APIKEY
  os.environ["OPENAI_API_KEY"] = openai_api_key



# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False
# PERSIST = st.secrets.app.persist

query = None

## getting the hostname by socket.gethostname() method
# hostname = socket.gethostname()
## getting the IP address using socket.gethostbyname() method
# ip_address = socket.gethostbyname(hostname)
# print(f"Hostname: {hostname}")
# print(f"IP Address: {ip_address}")

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  # loader = WebBaseLoader("https://www.growcentric.ro/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

# model = "gpt-4-1106-preview"
model = st.secrets.openai.model
chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model=model),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []

def generate_response(query):
    print(query['content'])
    result = chain({"question": query['content'], "chat_history": chat_history})

    return result['answer']

st.title(st.secrets.app.title) 
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": st.secrets.app.intro_message}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    openai.api_key = openai_api_key
    usermsg = {"role": "user", "content": prompt}
    st.session_state.messages.append(usermsg)
    st.chat_message("user").write(prompt)
    response = generate_response(usermsg)
    # msg = response.choices[0].message
    msg = {"role": "assistant", "content": response}
    print(msg['content'])
    st.session_state.messages.append(msg)
    # st.chat_message("assistant").write(msg)
    st.chat_message("assistant").write(msg['content'])