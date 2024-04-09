__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import json
import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import sys
import warnings
import pprint

from stores import DesignStore

# warnings.filterwarnings('ignore')

# Configuring OpenAI (GPT)
# print()
# ENV_PATH = sys.path[0]+'/andres.env'
# print("Reading OPENAI config:", ENV_PATH, load_dotenv(dotenv_path=Path(ENV_PATH)))
# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYSTEM_DESCRIPTION = './data/system_description.txt'
FUNCTIONAL_REQUIREMENTS = './data/requirements.json'
DESIGN_DECISIONS = './data/decisions.json'

def create_database(folder=None):
    # Load the system description
    system = None
    with open(SYSTEM_DESCRIPTION, 'r') as f:
        system_description = f.read()

    # Load a list of predefined requirements for the system
    requirements = None
    with open(FUNCTIONAL_REQUIREMENTS, 'r') as f:
        requirements = json.load(f)
    predefined_requirements = requirements['functional_requirements']

    # Load a list of predefined design decisions for the system
    decisions = None
    with open(DESIGN_DECISIONS, 'r') as f:
        decisions = json.load(f)
    predefined_decisions = decisions['design_decisions']

    system = dict()
    system['description'] = system_description
    system['id'] = requirements['system_name']

    dstore = DesignStore(create=True, system=system, requirements=predefined_requirements, path=folder, summarize=True)
    
    print("**Design decisions:", len(predefined_decisions))
    for dd in predefined_decisions:
        id = dd['id']
        description = dd['description']
        pattern = dd['pattern']
        requirements = dd['requirements']
        dstore.add_decision(id, description, requirements, pattern)
    
    return dstore


# --- Main program

print()
dstore = create_database()
print()



