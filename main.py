"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
import os


from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import GitLoader
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

def load_chain(documents):
    """Logic for loading the chain you want to use should go here."""
    if is_gpt4:
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"
    llm = ChatOpenAI(temperature=0.9, model_name=model, streaming=True, verbose=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap  = 20, length_function = len)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)   # db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever,max_tokens_limit=4096,memory=memory)
    return chain

def get_text():
    input_text = st.text_input("You: ", "what is this about?", key="input")
    return input_text

# From here down is all the StreamLit UI.
st.set_page_config(page_title="🔗😽ChatGIT", page_icon="🔗😽")
st.header("🔗😽 ChatGIT")

is_gpt4 = st.checkbox('Enable GPT4',help="With this it might get slower")
ask_button = ''

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

st.write('Please enter the repository url and the path. if needed, please specify the file type to treat. otherwise it will read all the documents')
with st.form(key='load'):
    url = st.text_input('clone url','https://github.com/hwchase17/langchain')
    path = st.text_input('repo path', './example_data/test_repo2/')
    branch = st.text_input('branch', 'master')
    extention = st.text_input('extention filter', '.py')
    load = st.form_submit_button('load')

if load:
    try:
        loader = GitLoader(
            clone_url=url,
            repo_path=path,
            branch=branch,
            file_filter=lambda file_path: file_path.endswith(extention))
    except:
        pass
    
    documents = loader.load()
    qa = load_chain(documents)
    user_input = get_text()
    ask_button = st.button('ask')
else:
    pass

language = st.selectbox('language',['English','日本語','Estonian'])

if ask_button:
    chat_history = []
    prefix = f'You are the best coding coach. please answer the question of the user. if possible, give some coffee examples so that they can understand easier.  please answer in {language}. User: '
    result = qa({"question": prefix + user_input, "chat_history": chat_history})
    st.session_state.past.append(user_input)
    st.session_state.generated.append(result['answer'])
    
if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
