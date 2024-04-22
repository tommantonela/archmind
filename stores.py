from typing import List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.retrievers import MergerRetriever

from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

import json
import pprint
import os
from tqdm import tqdm


class DesignStore:

    CHROMADB_PATH = "./system_chromadb"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, system:str=None, requirements:dict=None, create=False, path=None, summarize=False) -> None:

        path = path if path is not None else self.CHROMADB_PATH

        if summarize:
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDINGS_MODEL)
        self.persistent_client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
        
        if create:
            # existing_collections = self.persistent_client.list_collections()
            # if len(existing_collections) > 0:
            #     print("Deleting existing collections ...")
            #     for c in existing_collections:
            #         self.persistent_client.delete_collection(c.name)
            print("Deleting existing collections ...")
            self.persistent_client.reset()
            self.system_collection = self.persistent_client.get_or_create_collection("system", embedding_function=self.sentence_transformer_ef)
            self.requirements_collection = self.persistent_client.get_or_create_collection("requirements", embedding_function=self.sentence_transformer_ef)
            self.decisions_collection = self.persistent_client.get_or_create_collection("decisions", embedding_function=self.sentence_transformer_ef)

            # Add the documents here
            print("Creating collections and adding data ...")
            if system is not None:
                sys_id = system['id']
                sys_description = system['description']
                print("**System:", sys_id)
                # print(sys_description)
                self.system_collection.add(ids=[sys_id], documents=[sys_description], metadatas=[{"source": "original description"}])
                if summarize:
                    doc =  Document(page_content=sys_description)
                    summarize_chain = load_summarize_chain(self.llm, chain_type="stuff")
                    result = summarize_chain.invoke([doc])
                    print("summary:", result['output_text'])
                    self.system_collection.add(ids=[sys_id+"_summary"], documents=[result['output_text']], metadatas=[{"source": "summarized description"}])
            
            if requirements is not None:
                print("**Functional requirements:", len(requirements))
                # print(requirements)
                # pprint.pprint(requirements)
                all_ids = [r['id'] for r in requirements]
                all_docs = [r['description'] for r in requirements]
                
                # Store the patterns for each requirement
                all_patterns = [r['patterns'] for r in requirements]
                all_metadatas = DesignStore._convert_list_to_metadata(all_patterns, prefix="pattern")
                # print(all_metadatas)
                self.requirements_collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)

        else:
            print("Recovering existing collections ...")
            self.system_collection = self.persistent_client.get_or_create_collection("system", embedding_function=self.sentence_transformer_ef)
            self.requirements_collection = self.persistent_client.get_or_create_collection("requirements", embedding_function=self.sentence_transformer_ef)
            self.decisions_collection = self.persistent_client.get_or_create_collection("decisions", embedding_function=self.sentence_transformer_ef)
    
    @staticmethod
    def _convert_list_to_metadata(all_items:List, prefix):
        all_metadatas = []
        for dp_list in all_items:
            md = dict()
            dp_list = list(set(dp_list))
            for i, x in enumerate(dp_list):
                    md[prefix+"_"+str(i+1)] = x
            if len(md) > 0:
                all_metadatas.append(md)
            else:
                all_metadatas.append(None)
        return all_metadatas

    def get_system(self, sys_id=None, summary=False):
        if sys_id is None:
            result = self.system_collection.get()
        elif not summary:
            result = self.system_collection.get(ids=[sys_id], where={"source": "original description"})
        else:
            result = self.system_collection.get(ids=[sys_id+"_summary"], where={"source": "summarized description"})
        return result
    
    def get_requirement(self, id:str):
        result = self.requirements_collection.get(ids=[id])
        return result
    
    def get_patterns_for_requirement(self, id:str):
        result = self.requirements_collection.get(ids=[id])
        patterns = []
        if result['metadatas'][0] is not None:
            patterns = [p for k,p in result['metadatas'][0].items() if "pattern" in k]
        return patterns
    
    def get_requirements(self):
        result = self.requirements_collection.get()
        return result

    def get_decision(self, id:str):
        result = self.decisions_collection.get(ids=[id])
        return result
        
    def get_decisions(self):
        result = self.decisions_collection.get()
        return result

    def get_requirements_for_decision(self, id:str):
        result = self.decisions_collection.get(ids=[id])
        requirements = [p for k,p in result['metadatas'][0].items() if "requirement" in k]
        return requirements
    
    def add_requirement(self, id:str, description:str, candidate_patterns:List=None, update=False):
        # If the id already exists, it should override it in the collection
        result = self.requirements_collection.get(ids=[id])
        if len(result['ids']) > 0:
            print("Warning: requirement already exists, overwriting ...", id, result)
            update = True

        all_metadatas = None
        if candidate_patterns is not None:
            all_metadatas = DesignStore._convert_list_to_metadata([candidate_patterns], prefix="pattern")
        # print(candidate_patterns, all_metadatas)

        if not update:
            if all_metadatas is not None:
                self.requirements_collection.add(ids=[id], documents=[description], metadatas=all_metadatas)
            else: 
                self.requirements_collection.add(ids=[id], documents=[description])
        else:
            if all_metadatas is not None:
                self.requirements_collection.update(ids=[id], documents=[description], metadatas=all_metadatas)
            else:
                self.requirements_collection.update(ids=[id], documents=[description])

    def set_candidate_patterns_for_requirement(self, id:str, patterns:list):
        result = self.requirements_collection.get(ids=[id])
        if len(result['ids']) == 0:
            print("Error: requirement not found, cannot add patterns ...", id, result)
            return
        self.add_requirement(result['ids'][0], result['documents'][0], patterns, update=True)

    def add_decision(self, id:str, decision:str, requirements:List=None, main_pattern=None, update=False):
        # If the id already exists, it should override it in the collection
        result = self.decisions_collection.get(ids=[id])
        if len(result['ids']) > 0:
            print("Warning: decision already exists, overwriting ...", id, result)
            update = True

        all_metadatas = None
        if requirements is not None:
            all_metadatas = DesignStore._convert_list_to_metadata([requirements], prefix="requirement")
        if main_pattern is not None:
            if all_metadatas is None:
                all_metadatas = []
                all_metadatas.append({"pattern": main_pattern})
            else:
                all_metadatas[0]["pattern"] = main_pattern
        # print(requirements, all_metadatas)

        if not update:
            if all_metadatas is not None:
                self.decisions_collection.add(ids=[id], documents=[decision], metadatas=all_metadatas)
            else:
                self.decisions_collection.add(ids=[id], documents=[decision])
        else:
            if all_metadatas is not None:
                self.decisions_collection.update(ids=[id], documents=[decision], metadatas=all_metadatas)
            else:
                self.decisions_collection.update(ids=[id], documents=[decision])
    
    def set_requirements_for_decision(self, id:str, requirements:list):
        result = self.decisions_collection.get(ids=[id])
        if len(result['ids']) == 0:
            print("Error: decision not found, cannot add requirements ...", id, result)
            return
        self.add_decision(result['ids'][0], result['documents'][0], requirements, update=True)
    
    def search_related_decisions(self, decision:str, k:int=3, threshold=0.5, semantic_search=False):
        decisions = dict()
        target_decision = decision
        if not semantic_search:
            result = self.decisions_collection.get(ids=[target_decision])
            if len(result['ids']) > 0:
                target_decision = result['documents'][0]
            else:
                return (decisions)
        # Semantic search
        decs = self.decisions_collection.query(query_texts=[target_decision], n_results=k) 
        # print(decs)
        for d, distance, description in zip(decs['ids'][0], decs['distances'][0], decs['documents'][0]):
            if (target_decision != description) and (distance <= threshold) and (distance > 0.0):
                decisions[d] = description
        return decisions
    
    def search_decisions_for_requirement(self, requirement: str, k:int=3, semantic_search=False, threshold=0.5):
        decisions = dict()
        if not semantic_search: # requirement is considered as an id
            docs = self.get_decisions()
            for d, metadata, description in zip(docs['ids'], docs['metadatas'], docs['documents']):
                for k, v in metadata.items():
                    if ("requirement" in k) and (requirement == v): 
                        decisions[d] = description
            return decisions
        else: # requirement is used for semantic search
            reqs = self.requirements_collection.query(query_texts=[requirement], n_results=k)
            # print(reqs)
            for r, distance in zip(reqs['ids'][0], reqs['distances'][0]):
                if (distance <= threshold) and (distance > 0.0):
                    # print("Checking", r, distance)
                    decisions_for_r = self.search_decisions_for_requirement(r, k, semantic_search=False)
                    decisions.update(decisions_for_r)
            return decisions
    
    def search(self, query:str, collection:str, k:int=3):
        if (collection != "requirements") and (collection != "decisions"):
           print("Error: collection not found ...", collection)
           return []

        langchain_embedding_function = SentenceTransformerEmbeddings(model_name=DesignStore.EMBEDDINGS_MODEL)
        langchain_chroma = Chroma(client=self.persistent_client,
            collection_name=collection,
            embedding_function=langchain_embedding_function,
        )
        print("There are", langchain_chroma._collection.count(), "items in the", collection, "collection")

        docs = langchain_chroma.similarity_search(query, k=k) # It relies on Langchain wrapper
        return docs


class KnowledgeStore:

    CHROMADB_PATH = "./patterns_chromadb"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    def __init__(self, create=False, path=None) -> None:

        self.db_path = path if path is not None else self.CHROMADB_PATH

        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDINGS_MODEL)
        self.persistent_client = chromadb.PersistentClient(path=self.db_path, settings=Settings(allow_reset=True))

        self.embeddings = SentenceTransformerEmbeddings(model_name=KnowledgeStore.EMBEDDINGS_MODEL) # Local embeddings

        if create:
            print("Deleting existing collections ...")
            self.persistent_client.reset()
        
        self.dpatterns_collection = self.persistent_client.get_or_create_collection("design_patterns", embedding_function=self.sentence_transformer_ef)
        self.styles_collection = self.persistent_client.get_or_create_collection("architectural_styles", embedding_function=self.sentence_transformer_ef)
        self.microservices_collection = self.persistent_client.get_or_create_collection("microservice_patterns", embedding_function=self.sentence_transformer_ef)

        self.dpatterns_vectordb = Chroma(collection_name="design_patterns", persist_directory=self.db_path, embedding_function=self.embeddings)  
        print("There are", self.dpatterns_vectordb._collection.count(), "chunks in the design patterns collection")

        self.styles_vectordb = Chroma(collection_name="architectural_styles", persist_directory=self.db_path, embedding_function=self.embeddings)  
        print("There are", self.styles_vectordb._collection.count(), "chunks in the architectural styles collection")
        # db_records = self.styles_vectordb.get()
        # print("Loaded", len(db_records['ids']), "records from the architectural styles collection")

        self.microservices_vectordb = Chroma(collection_name="microservice_patterns", persist_directory=self.db_path, embedding_function=self.embeddings)  
        print("There are", self.microservices_vectordb._collection.count(), "chunks in the microservice patterns collection")

    @staticmethod
    def _process_pdf_batch(pdf_files):
        batch_docs = []
        for pdf_file_path in tqdm(pdf_files, "PDFs"):
            pdf_loader = PyPDFLoader(pdf_file_path)
            batch_docs.extend(pdf_loader.load())
        
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=KnowledgeStore.CHUNK_SIZE, chunk_overlap=KnowledgeStore.CHUNK_OVERLAP)
        # pdf_chunks = text_splitter.split_documents(batch_docs)
        # print(len(pdf_chunks), "chunks")

        # text_splitter = SemanticChunker(OpenAIEmbeddings())
        text_splitter = SemanticChunker(HuggingFaceEmbeddings())
        # text_splitter = SemanticChunker(model_name=SentenceTransformerEmbeddings(KnowledgeStore.EMBEDDINGS_MODEL))
        pdf_chunks = text_splitter.split_documents(batch_docs)

        return pdf_chunks
    
    def ingest_architectural_patterns(self, pdf_files:List[str]):
        print("Ingesting PDFs for architectural patterns ...")
        pdf_chunks = KnowledgeStore._process_pdf_batch(pdf_files)
        print(len(pdf_chunks), "chunks")

        self.styles_vectordb = Chroma.from_documents(pdf_chunks, embedding=self.embeddings, persist_directory=self.db_path, collection_name="architectural_styles")
    
    def ingest_design_patterns(self, pdf_files:List[str]):
        print("Ingesting PDFs for design patterns ...")
        pdf_chunks = KnowledgeStore._process_pdf_batch(pdf_files)
        print(len(pdf_chunks), "chunks")

        self.styles_vectordb = Chroma.from_documents(pdf_chunks, embedding=self.embeddings, persist_directory=self.db_path, collection_name="design_patterns")

    def ingest_microservice_patterns(self, pdf_files:List[str]):
        print("Ingesting PDFs for microservice patterns ...")
        pdf_chunks = KnowledgeStore._process_pdf_batch(pdf_files)
        print(len(pdf_chunks), "chunks")

        self.styles_vectordb = Chroma.from_documents(pdf_chunks, embedding=self.embeddings, persist_directory=self.db_path, collection_name="microservice_patterns")

    def search(self, query:str, collection:str, k:int=3):
        if collection == "architectural_styles":
            vectordb = self.styles_vectordb
        elif collection == "design_patterns":
            vectordb = self.dpatterns_vectordb
        elif collection == "microservice_patterns":
            vectordb = self.microservices_vectordb
        else:
            print("Error: collection not found ...", collection)
            return []

        docs = vectordb.similarity_search(query, k=k) # It relies on Langchain wrapper
        return docs
    
    def _get_database(self, collection:str):
        if collection == "architectural_styles":
            vectordb = self.styles_vectordb
        elif collection == "design_patterns":
            vectordb = self.dpatterns_vectordb
        elif collection == "microservice_patterns":
            vectordb = self.microservices_vectordb
        else:
            print("Error: collection not found ...", collection)
            return None
        
        return vectordb
    
    def get_retriever(self, collection:str=None, threshold:float=0.5):
        retriever = None
        if collection is None:
            print("Warning: collection not found ...", collection)
            return None
        
        if collection == 'all':
            print("Creating LOTR ...")
            dp_retriever = self.get_retriever('design_patterns', threshold)
            ms_retriever = self.get_retriever('microservice_patterns', threshold)
            as_retriever = self.get_retriever('architectural_styles', threshold)
            retriever = MergerRetriever(retrievers=[dp_retriever, ms_retriever, as_retriever])
        else:
            chroma_db = self._get_database(collection=collection)
            # retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K}) 
            retriever = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": threshold}) 
        
        return retriever
