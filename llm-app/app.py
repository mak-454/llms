# streamlit_app.py

import streamlit as st
import requests
import json

PRELOADED_LLM_MODELS = {
    "GPTJ6B": "EleutherAI/gpt-j-6B",
    "GPTJT6B": "togethercomputer/GPT-JT-6B-v1",
    "BioGPT": "microsoft/biogpt",
    "BioGPT-Large": "microsoft/BioGPT-Large",
    "BioGPT-Large-PubMedQA": "microsoft/BioGPT-Large-PubMedQA",
    "BLOOM": "bigscience/bloom",
    "GPT-NEO": "EleutherAI/gpt-neo-2.7B",
}

DEPLOYMENT_ENDPOINT = "http://raycluster-llm-head-svc:8000"

MODELS_PREFIX = {
        "BioGPT": "/biogpt",
        "BioGPT-Large": "/biogpt-large",
        "BioGPT-Large-PubMedQA": "/biogpt-large-pubmedqa",
        "GPTJT6B": "/gptjt6b",
        "GPTJ6B": "/gptj6b",
        "BLOOM": "/bloom",
        "GPT-NEO": "/gpt-neo",
}

biogpt_pubmeqa = '["question: Should chest wall irradiation be included after mastectomy and negative node breast cancer? context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only. Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998. Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance. answer: Post-mastectomy radiotherapy should be discussed for a sub-group of node-negative patients with predictors factors of local failure such as age<or = 40 years and larger tumour size. target: the answer to the question given the context is"]'

gptjt6b = "In a shocking finding, scientists discovered a herd of unicorns living in a remote,  previously unexplored valley, in the Andes Mountains. Even more surprising to the  researchers was the fact that the unicorns spoke perfect English."

gptj6b = "Zoe Kwan is a 20-year old singer and songwriter who has taken Hong Kong’s music scene by storm."

biogpt = "A 65-year-old female patient with a past medical history of"

DEFAULT_TEXTS = {
        "BioGPT": biogpt,
        "BioGPT-Large": biogpt,
        "BioGPT-Large-PubMedQA": biogpt_pubmeqa,
        "GPTJT6B": gptjt6b,
        "GPTJ6B": gptj6b,
        "BLOOM": "",
        "GPT-NEO": ""
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


def send_prompt(settings, prompt):
    llm = settings['model']
    max_length = settings.get('answer_length', 1)
    temperature = settings.get('temperature', 0.0)
    prefix = MODELS_PREFIX[llm]
    url = DEPLOYMENT_ENDPOINT + prefix
    if llm == "BioGPT-Large-PubMedQA":
        prompt = json.loads(prompt)
        #st.write(prompt)
    data = {"prompt": prompt, "max_length": max_length, "temperature": temperature}
    resp = requests.post(url, json=data)
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
        sidebar["answer_length"] = st.slider(
                "Answer Length",
                min_value = 1,
                max_value = 200,
                value=100,
                )
        if sidebar["model"] == "GPTJT6B" or sidebar["model"] == "GPTJ6B":
            sidebar["temperature"] = st.slider(
                    "Temperature",
                    min_value= 0.1,
                    max_value = 1.0,
                    value = 0.7
                    )
        else:
            sidebar["num_sequences"] = st.slider(
                    "Num of sequences",
                    min_value= 1,
                    max_value = 10,
                    value = 1
                    )


    st.title("LLM Tester")
    st.caption("Tool to test LLMs deployed on an internal cluster")
    #st.caption("Input your prompt ..")

    
    default_prompt = DEFAULT_TEXTS[sidebar["model"]]
    prompt = st.text_area(
        label="Input your prompt ..",
        value=default_prompt,
        height=200,
        )

    st.session_state['prompt_ta'] = prompt
    if st.button("Submit", disabled=st.session_state.get("disabled", False), on_click=send_prompt, args=(sidebar,prompt)):
        answer = st.text_area(
            label="Answer from LLM",
            value=st.session_state["answer"],
            max_chars=10000,
            height=300,
            )
        #st.write(answer)

    #st.write("Here goes your normal Streamlit app...")
    #st.button("Click me")
