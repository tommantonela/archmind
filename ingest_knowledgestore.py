__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from stores import KnowledgeStore

# warnings.filterwarnings('ignore')

DESIGN_PATTERNS_DOCS_PATH = "./docs/design-patterns"
MICROSERVICE_PATTERNS_DOCS_PATH = "./docs/microservice-patterns"
ARCHITECTURE_STYLES_DOCS_PATH = "./docs/architectural-styles"

def create_database(folder=None):
    kstore = KnowledgeStore(create=True, path=folder)

    # Ingestion of design patterns
    pdf_files_to_process = []
    for root, dirs, files in os.walk(DESIGN_PATTERNS_DOCS_PATH):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    for file in pdf_files_to_process:
        print(file)
    print(len(pdf_files_to_process), "pdfs to process")
    kstore.ingest_design_patterns(pdf_files_to_process)

    # Ingestion of microservice patterns
    pdf_files_to_process = []
    for root, dirs, files in os.walk(MICROSERVICE_PATTERNS_DOCS_PATH):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    for file in pdf_files_to_process:
        print(file)
    print(len(pdf_files_to_process), "pdfs to process")
    kstore.ingest_microservice_patterns(pdf_files_to_process)
    
    # Ingestion of architectural patterns
    pdf_files_to_process = []
    for root, dirs, files in os.walk(ARCHITECTURE_STYLES_DOCS_PATH):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    for file in pdf_files_to_process:
        print(file)
    print(len(pdf_files_to_process), "pdfs to process")
    kstore.ingest_architectural_patterns(pdf_files_to_process)

    return kstore


# --- Main program

print()
kstore = create_database()
print()