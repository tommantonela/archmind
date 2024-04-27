from stores import KnowledgeStore, DesignStore
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import warnings
import json

from assistants import PatternAssessmentCopilot
from assistants import PatternAssessment

# --- Main program ---
            
warnings.filterwarnings('ignore')
                        
# Configuring OpenAI (GPT)
print()
ENV_PATH = sys.path[0]+'/andres.env'
print("Reading OPENAI config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0).bind_tools([PatternAssessment])
use_openai_functions = True

chroma_db1 = "./patterns_chromadb"
kstore = KnowledgeStore(path=chroma_db1)
chroma_db2 = "./system_chromadb"
dstore = DesignStore(path=chroma_db2)

copilot = PatternAssessmentCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
# copilot.configure_retriever('all', openai_functions=use_openai_functions)
# copilot.configure_retriever('design_patterns', openai_functions=use_openai_functions)
# copilot.configure_retriever('microservice_patterns', openai_functions=use_openai_functions)
# copilot.configure_retriever('architectural_styles', openai_functions=use_openai_functions)
copilot.configure_retriever(collection=None, openai_functions=use_openai_functions)

# copilot.launch(openai_functions=use_openai_functions)

EXPORT_FOLDER = './output/assessments.json'

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

print("-"*10)
all_results = []
for d,r in REQUIREMENTS.items():
    result_dict = copilot.process_requirements_decision(r,d)
    if result_dict is not None:
        # print(d,result_dict['decision'])
        result_dict['decision'] = d
        all_results.append(result_dict)
    else:
        print(d, "Nothing generated for", r, d, result_dict)
    print("-"*10)
with open(EXPORT_FOLDER, 'w') as fout:
    json.dump(all_results , fout, indent=4)
