from stores import KnowledgeStore, DesignStore
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import warnings
import json

from assistants import ADRWriterCopilot

# --- Main program ---
            
warnings.filterwarnings('ignore')
                        
# Configuring OpenAI (GPT)
print()
ENV_PATH = sys.path[0]+'/andres.env'
print("Reading OPENAI config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

chroma_db1 = "./patterns_chromadb"
kstore = KnowledgeStore(path=chroma_db1)
chroma_db2 = "./system_chromadb"
dstore = DesignStore(path=chroma_db2)

copilot = ADRWriterCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
# copilot.configure_retriever('all')
# copilot.configure_retriever('design_patterns')
# copilot.configure_retriever('microservice_patterns')
# copilot.configure_retriever('architectural_styles')
copilot.configure_retriever(collection=None)

# copilot.launch()

EXPORT_FOLDER = './output/'
ASSESSMENTS_INPUT_FOLDER = './data/rag/assessments-rag.json'
RANKINGS_INPUT_FOLDER = './data/rag/rankings-rag.json'


REQUIREMENTS = {
    "DD1": "RF1",
    "DD2": "[RF3, RF3.3]",
    "DD3": "RF4",
    "DD4": "RF4",
    "DD5": "[RF2, RF3.1, RF3.2]",
    "DD6": "RF8",
    "DD7": "[RF5, RF6]",
    "DD8": "RF3.3",
    "DD9": "RF6",
    "DD10": "RF6",
    "DD11": "RF5",
    "DD12": "RF7.1",
    "DD13": "RF1",
    "DD14": "RF7",
    "DD15": "RF3.2",
    "DD16": "[RF2.1, RF3.1]"
}

def convert_assessment_to_string(assessment):
    s = ""
    for k,v in assessment.items():
        if (k != "decision") & (k != "appropriateness"):
            if k == 'clarifying_questions':
                k = "Clarifying questions: "
            elif k == 'assumptions_and_constraints':
                k = "Assumptions and constraints: "
            elif k == 'qa_consequences':
                k = "Consequences on quality attributes: "
            elif k == 'risks_and_tradeoffs':
                k = "Risks and tradeoffs: "
            elif k == 'followup_decisions':
                k = "Follow-up decisions: "
            s += k+" "+v+"\n"
    return s

def convert_ranking_to_string(ranking):
    s = ""
    for idx, r in enumerate(ranking['ranking']):
        # print((idx+1), type(r))
        if isinstance(r, dict):
            s += str(idx+1) + ". "
            s += r['pattern_name'] + "\n"
            s += r['description'] + "\n"
            s += "Pros: " + r['pros'] + "\n"
            s += "Cons: " + r['cons'] + "\n\n"
    s = s + "\n"
    return s

with open(ASSESSMENTS_INPUT_FOLDER) as f:
    assessments = json.load(f)
assessments_dict = {a['decision']: convert_assessment_to_string(a) for a in assessments}

with open(RANKINGS_INPUT_FOLDER) as f:
    rankings = json.load(f)
rankings_dict = {r['decision']: convert_ranking_to_string(r) for r in rankings}

print("-"*10)
all_results = []
for d,r in REQUIREMENTS.items():
    a = assessments_dict[d]
    # print(a)
    rk = rankings_dict[d]
    # print(r)
    adr = copilot.process_requirements_decision_analysis(r, d, a, rk)
    if adr is not None:
        filename = EXPORT_FOLDER + "adr_"+d.lower()+ ".md"
        print("Generating ADR file:", filename)
        with open(filename, 'w') as fout:
            fout.write(adr)
    else:
        print("Nothing generated for", r, d)
    print("-"*10)
