import requests

#url = "http://127.0.0.1:8000/gptjt6b"
url = "http://127.0.0.1:8000/biogpt"
url = "http://127.0.0.1:8000/probe"
#data =   "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
data = " Cancer is"
data = "biogpt"
#data = ["question: Should chest wall irradiation be included after mastectomy and negative node breast cancer? context: This study aims to evaluate local failure patterns in node negative breast cancer patients treated with post-mastectomy radiotherapy including internal mammary chain only. Retrospective analysis of 92 internal or central-breast node-negative tumours with mastectomy and external irradiation of the internal mammary chain at the dose of 50 Gy, from 1994 to 1998. Local recurrence rate was 5 % (five cases). Recurrence sites were the operative scare and chest wall. Factors associated with increased risk of local failure were age<or = 40 years and tumour size greater than 20mm, without statistical significance. answer: Post-mastectomy radiotherapy should be discussed for a sub-group of node-negative patients with predictors factors of local failure such as age<or = 40 years and larger tumour size. target: the answer to the question given the context is"]
resp = requests.post(url, json=data)
print(resp.content)
