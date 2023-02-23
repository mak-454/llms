import streamlit as st

import pandas as pd
import io

csvstr = """
model,size,memory,#gpus,latency
microsoft/biogpt,1.56GB,1G,1,UNKNOWN
microsoft/BioGPT-Large,6.29GB,5G,1,UNKNOWN
microsoft/BioGPT-Large-PubMedQA,6.29GB,5G,1,UNKNOWN
togethercomputer/GPT-JT-6B-v1,12.2GB,10G,2,UNKNOWN
EleutherAI/gpt-j-6B,12.2GB,10G,2,UNKNOWN
"""

df = pd.read_csv(io.StringIO(csvstr), sep=",")

# style
th_props = [
  ('font-size', '14px'),
  ('text-align', 'center'),
  ('font-weight', 'bold'),
  ('color', '#6d6d6d'),
  ('background-color', '#f7ffff')
  ]

td_props = [
  ('font-size', '12px')
  ]

styles = [
  dict(selector="th", props=th_props),
  dict(selector="td", props=td_props)
  ]

# table
df2=df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)

st.markdown("Single AWS instance of type `p2.8xlarge` 8 K80 GPUs")
st.table(df2)
