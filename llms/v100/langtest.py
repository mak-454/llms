import time
from langchain.llms import DKubeAI

st = time.time()
llm = DKubeAI(model_kwargs={"max_new_tokens":100, "temperature":0.7, "repetition_penalty": 1.2}, dkubeai_ep="http://ade5739a2b22a44708d0225feaeb06eb-1064712555.us-east-2.elb.amazonaws.com:8000", repo_id="togethercomputer/GPT-JT-6B-v1")
text = """Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.
Possible labels:
1. hate speech
2. not hate speech

Tweet: HOW REFRESHING! In South Korea, there is no such thing as 'political correctness" when it comes to dealing with Muslim refugee wannabes via @user
Label: hate speech

Tweet: New to Twitter-- any men on here know what the process is to get #verified?
Label: not hate speech

Tweet: Dont worry @user you are and will always be the most hysterical woman.
Label:"""
#text = "Zoe Kwan is a 20-year old singer and songwriter who has taken Hong Kong’s music scene by storm."
text = """Context information is below.
doi: 10.1111/nph.14926 pmid: 29568489 pmcid: PMC5850084 title: Ten steps to get started in Genome Assembly and Annotation. journal: F1000Research authors: Victoria Dominguez Del Angel; Erik Hjerde; Lieven Sterck; Salvadors Capella-Gutierrez; Cederic Notredame; Olga Vinnere Pettersson; Joelle Amselem; Laurent Bouri; Stephanie Bocs; Christophe Klopp; Jean-Francois Gibrat; Anna Vlasova; Brane L Leskosek; Lucile Soler; Mahesh Binzer-Panchal; Henrik Lantz year: 2018 keywords: Annotation;Assembly;DNA;FAIR;Genome;NGS;Workflows section: 4. Estimate the necessary computational resources

To succeed in a genome assembly and annotation project you need to have sufficient compute resources. The resource demands are different between assembly and annotation, and different tools also have very different requirements, but some generalities can be observed (for examples, see Table 1). For genome assembly, running times and memory requirements will increase with the amount of data. As more data is needed for large genomes, there is thus also a correlation between genome size and running time/memory requirements. Only a small subset of available assembly programs can distribute the assembly into several processes and run them in parallel on several compute nodes. Tools that cannot do this tend to require a lot of memory on a single node, while programs that can split the process need less memory in each individual node, but do on the other work most efficiently when several nodes are available. It is therefore important to select the proper assembly tools early in a project, and make sure that there are enough available compute resources of the right type to run these tools. Annotation has a different profile when it comes to computer resource use compared to assembly. When external data such as RNA-seq or protein sequences are used (something that is strongly recommended), mapping these sequences to the genome is a major part of the annotation process. Mapping is computationally intense, and it is highly preferable to use annotation tools that can run on several nodes in parallel. Regarding storage, usually no extra consideration needs to be taken for assembly or annotation projects compared to other NGS projects. Intermediate files are often much larger than the final results, but can often be safely deleted once the run is finished.
Given the context information and not prior knowledge, answer the question: What methods can be used to annotate genomics samples?"""

answer= llm(text)
et = time.time()

print(answer)
print("-------Elapsed time:", et-st, "------------")
