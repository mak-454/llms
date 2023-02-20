# streamlit_app.py

import streamlit as st
import requests
import json

PRELOADED_LLM_MODELS = {
    "BioGPT": "microsoft/biogpt",
    "BioGPT-Large": "microsoft/BioGPT-Large",
    "BioGPT-Large-PubMedQA": "microsoft/BioGPT-Large-PubMedQA",
    "GPTJT6B": "togethercomputer/GPT-JT-6B-v1",
    "GPT-J-6B": "EleutherAI/gpt-j-6B",
    "BLOOM": "bigscience/bloom",
    "GPT-NEO": "EleutherAI/gpt-neo-2.7B",
}

DEPLOYMENT_ENDPOINT = "http://raycluster-llm-head-svc:8000"

MODELS_PREFIX = {
        "BioGPT": "/biogpt",
        "BioGPT-Large": "/biogpt-large",
        "BioGPT-Large-PubMedQA": "/biogpt-large-pubmedqa",
        "GPTJT6B": "/gptjt6b",
        "GPT-J-6B": "/gpt6jb",
        "BLOOM": "/bloom",
        "GPT-NEO": "/gpt-neo",
}


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def change_selection():
    #st.info(f"Selected LLM: {st.session_state.choice}")
    choice = st.session_state.choice
    llm = choice.lower()
    url = DEPLOYMENT_ENDPOINT + "/probe"
    try:
        resp = requests.post(url, json=llm)
        if resp.status_code == 200:
            probe = json.loads(resp.content)
            if probe == "deployed":
                st.session_state["disabled"] = False
                st.info(f"LLM {choice} is deployed and HEALTHY")
            else:
                st.error(f"LLM {choice} is not deployed or UNHEALTHY, contact adminstrator")
        else:
            st.error(f"LLM {choice} is not deployed or UNHEALTHY, contact adminstrator")
    except Exception as exc:
            st.error(f"LLM {choice} is not deployed or UNHEALTHY, contact adminstrator")

def post_question(settings, prompt):
    llm = settings['model']
    prefix = MODELS_PREFIX[llm]
    url = DEPLOYMENT_ENDPOINT + prefix
    resp = requests.post(url, json=prompt)
    if resp.status_code == 200:
        answer = json.loads(resp.content)
    else:
        answer = f"api failed with {resp.status_code}"
    st.session_state["answer"] = answer
    print(answer)

if check_password():
    sidebar = {}
    st.session_state["disabled"] = False
    with st.sidebar:
        sidebar["model"] = st.radio(
            "Choose the LLM",
            tuple(k for k in PRELOADED_LLM_MODELS.keys()),
            on_change=change_selection,
            key="choice",
            help="Select the LLM model to test.",
            )
    st.title("LLM Tester")
    st.caption("Input your prompt ..")

    default_prompt = "COVID19 is ..."
    prompt = st.text_area(
        label="Input your prompt",
        value=default_prompt,
        )

    if st.button("Submit", disabled=st.session_state.get("disabled", False), on_click=post_question, args=(sidebar,prompt)):
        answer = st.text_area(
            label="Answer from LLM",
            value=st.session_state["answer"],
            max_chars=10000,
            )
        #st.write(answer)

    #st.write("Here goes your normal Streamlit app...")
    #st.button("Click me")
