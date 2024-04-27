
from stores import KnowledgeStore, DesignStore
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import warnings

from assistants import PatternSuggestionCopilot
from assistants import PatternRanking, PatternAnalysis

import json

# --- Main program ---
            
warnings.filterwarnings('ignore')
                        
# Configuring OpenAI (GPT)
print()
ENV_PATH = sys.path[0]+'/andres.env'
print("Reading OPENAI config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) #, verbose=True)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0).bind_tools([PatternRanking, PatternAnalysis])
use_openai_functions = True

# chroma_db1 = "./patterns_semantic_chromadb" 
chroma_db1 = "./patterns_chromadb"
kstore = KnowledgeStore(path=chroma_db1)
chroma_db2 = "./system_chromadb"
dstore = DesignStore(path=chroma_db2)

copilot = PatternSuggestionCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
# copilot.configure_retriever('all', openai_functions=use_openai_functions)
# copilot.configure_retriever('design_patterns', openai_functions=use_openai_functions)
# copilot.configure_retriever('microservice_patterns', openai_functions=use_openai_functions)
# copilot.configure_retriever('architectural_styles', openai_functions=use_openai_functions)
copilot.configure_retriever(collection=None, openai_functions=use_openai_functions)

#copilot.launch(openai_functions=use_openai_functions)

RANKINGS_EXPORT_FOLDER = './output/rankings.json'
DECISIONS_EXPORT_FOLDER = './output/decisions.json'

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
all_rankings = []
for d,r in REQUIREMENTS.items():
    result_dict = copilot.process_requirements(r)
    if result_dict is not None:
        print(d,result_dict['decision'])
        result_dict['decision'] = d
        all_rankings.append(result_dict)
    else:
        print(d, "Nothing generated for", r, result_dict)
    print("-"*10)

all_decisions = []
for r in all_rankings:
    dec = dict()
    dec['id'] = r['decision']
    dec['description'] = r['description']
    dec['pattern'] = ""
    dec['requirements'] = r['requirement']
    all_decisions.append(dec)

with open(RANKINGS_EXPORT_FOLDER, 'w') as fout:
    json.dump(all_rankings , fout, indent=4)

with open(DECISIONS_EXPORT_FOLDER, 'w') as fout:
    json.dump(all_decisions, fout, indent=4)

