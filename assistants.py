__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.exceptions import OutputParserException
from langchain.schema.retriever import BaseRetriever
from langchain.docstore.document import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.callbacks import BaseCallbackHandler

from typing import TYPE_CHECKING, Any, Dict, List, Optional 

import pprint
import logging

class CustomHandler(BaseCallbackHandler):
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        formatted_prompts = "\n".join(prompts)
        #_log.info(f"Prompt:\n{formatted_prompts}")
        print(f"Prompt:\n{formatted_prompts}")


class DummyRetriever(BaseRetriever):

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return [] # [Document(page_content="Dummy document", metadata={"dummy": "metadata"})]


class BaseCopilot:

    TOP_K = 3
    THRESHOLD = 0.5
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

    def configure_retriever(self, collection:str=None, threshold=None) -> None:
        pass

    def launch(self) -> None:
        pass

    def reset(self) -> None:
        pass

    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _pretty_print_docs(docs):
        print(
            f"\n{'-' * 100}\n".join(
            [f"Document {i+1}:\n\n" + d.page_content +"\n"+str(d.metadata) for i, d in enumerate(docs)]
            )
        )
    
    def get_system_summary(self) -> str:
        return self.summary
        
    def set_system_summary(self,new_summary):
        self.summary = new_summary
    
    def get_requirement(self) -> str:
        return self.requirement
    
    def get_decision(self) -> str:
        return self.decision
    
    def set_decision(self, decision: str):
        self.decision = decision
    
    def set_requirement(self, requirement: str):
        self.requirement = requirement
    
    def get_patterns(self) -> list:
        return self.patterns
    
    def set_patterns(self, patterns: list):
        self.patterns = patterns
    
    def get_requirements(self):
        result = self.sys_store.get_requirements()
        ids = result['ids']
        docs = result['documents']
        if len(ids) > 0:
            return dict(zip(ids, docs))
        else:
            return None
    
    def get_decisions(self):
        result = self.sys_store.get_decisions()
        ids = result['ids']
        docs = result['documents']
        # print(result)
        if len(ids) > 0:
            return dict(zip(ids, docs))
        else:
            return None
    
    # def get_requirement_db(self, query): 
    #     result = self.sys_store.get_requirement(query)
    #     print(result)
    #     if len(result['documents']) != 0:
    #         return result['documents'][0]
    #     return None

    def fetch_requirement(self, query):
        result = self.sys_store.get_requirement(query)
        # print(result)

        patterns = None
        if len(result['documents']) == 0:
            print("- requirement not found, proceeding with original query")
            requirement = None # query
            patterns = []
        else:    
            print(query, "- requirement found:", result['documents'][0])
            requirement = result['documents'][0]
            patterns = self.sys_store.get_patterns_for_requirement(result['ids'][0])
        
        return requirement, patterns
    
    def fetch_decision(self, query):
        result = self.sys_store.get_decision(query)
        # print(result)

        requirements = None
        if len(result['documents']) == 0:
            print("- decision not found, proceeding with original query")
            decision = None # query
            requirements = []
        else:    
            print(query, "- decision found:", result['documents'][0])
            decision = result['documents'][0]
            requirements = self.sys_store.get_requirements_for_decision(result['ids'][0])
        
        return decision, requirements

    def get_decisions_by_requirement(self):
        all_reqs = self.get_requirements()
        result = dict()
        for r in all_reqs.keys():
            decisions = list(self.sys_store.search_decisions_for_requirement(r).values())
            if len(decisions) > 0:
                result[r] = decisions
        return result


class PatternsQACopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE_RAG = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to answer the question at the end.
    If the context is empty or you don't know the answer, then just say that you don't know, don't try to make up an answer.
    Use {n} sentences maximum and keep your answer as concise as possible.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""

    SYSTEM_PROMPT_TEMPLATE_ZEROSHOT = """You are an experienced software architect that assists a novice developer 
    to design a system. Answer the question asked by the novice developer at the end.
    If you don't know the answer, then just say that you don't know, don't try to make up an answer.
    Use {n} sentences maximum and keep your answer as concise as possible.

    Context:
    {context}

    Question: {question}

    Helpful Answer:"""

    CONTEXTUALIZED_COPILOT_PROMPT = """Given a chat history and the latest developer's question
    which might reference context in the chat history, formulate a standalone question
    that can be understood without the chat history. Do NOT answer the question,
    just reformulate it if needed and otherwise return it as is."""

    def __init__(self, store, llm, show_prompts=False) -> None:
        self.llm = llm
        self.store = store
        self.embeddings = SentenceTransformerEmbeddings(model_name=PatternsQACopilot.EMBEDDINGS_MODEL)
        self.chat_history = ChatMessageHistory()
        self.target = None

        self.config = {}
        if show_prompts:
            self.config = {"callbacks": [CustomHandler()]}

        self.contextualized_qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PatternsQACopilot.CONTEXTUALIZED_COPILOT_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.contextualized_qa_chain = self.contextualized_qa_prompt | llm | StrOutputParser()
    
    def configure_retriever(self, collection:str=None, threshold=None):
        if threshold is None:
            threshold = PatternsQACopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            prompt = PatternsQACopilot.SYSTEM_PROMPT_TEMPLATE_ZEROSHOT
        else:
            retriever = self.store.get_retriever(collection=collection, threshold=threshold)
            prompt = PatternsQACopilot.SYSTEM_PROMPT_TEMPLATE_RAG
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
        
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)

        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.rag_chain = (
            RunnablePassthrough.assign(
            context=self._contextualized_question | self.compression_retriever | PatternsQACopilot._format_docs)
            | self.qa_prompt
            | self.llm
        )
    
    def _contextualized_question(self, input: dict):
        if input.get("chat_history"):
            return self.contextualized_qa_chain
        else:
            return input["question"]
    
    def clear_chat_history(self):
        self.chat_history.clear()
    
    def run(self, query: str, history=True, n=4):

        retrieved_docs = self.compression_retriever.get_relevant_documents(query) 
        
        if (self.target is None) or (len(retrieved_docs) > 0): 
            ai_msg = self.rag_chain.invoke({'question': query, 'n': n, 
                                            'chat_history': self.chat_history.messages}, 
                                            config=self.config)
        else:
            ai_msg = AIMessage(content="I don't know")
        
        if history:
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(ai_msg)
        
        return ai_msg, retrieved_docs
    
    def launch(self):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            query = input('>>Prompt: ')
            # give us a way to exit the script
            if query == "exit" or query == "quit" or query == "q":
                print('Bye!')
                print()
                # sys.exit()
                return
        
            ai_msg, retrieved_docs = self.run(query)
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression)")
            # pretty_print_docs(retrieved_docs)
            print()

            print('>>Answer: ', ai_msg.content)
            print()


class PatternSuggestionCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, then just say return an empty list, don't try to make up an answer.
    """

    SELECTION_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    The question at the end is about a requirement that this software system must fulfill. 
    The requirement itself is enclosed by backticks. Remember to focus on the requirement.
    Answer with a plain list of patterns that might be applicable for the requirement. 
    For each pattern, assess if the pattern is related to the additional information provided. 
    If a pattern is not related to the additional information, do not include the pattern in the list.
    If the part with additional information is empty, just return an empty list.
    If the context is empty, just return an empty list.
    Return the patterns in JSON format. Only include the names of the patterns in the JSON structure, nothing else.
    Double-check if the requirement is clear and unambiguous for the provided context.
    Do not change or alter the original requirement in any way.
    If the requirement does not make sense for the context or is unclear or ambiguous, just return an empty list,
    do not try to provide irrelevant patterns.

    Context: {system_context}

    Additional information: 
    {pattern_context}

    Question: What patterns or tactics are applicable for the requirement: ``{requirement}``?

    Helpful answer:"""

    SELECTION_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed.
    The question at the end is about a requirement that this software system must fulfill. 
    The requirement itself is enclosed by backticks. Remember to focus on the requirement.
    Answer with a plain list of patterns that might be applicable for the requirement. 
    Return the patterns in JSON format. Only include the names of the patterns in the JSON structure, nothing else.
    If the context is empty, just return an empty list,
    Double-check if the requirement is clear and unambiguous for the provided context.
    Do not change or alter the original requirement in any way.
    If the requirement does not make sense for the context or is unclear or ambiguous, just return an empty list,
    do not try to provide irrelevant patterns.

    Context: {system_context}

    Question: What patterns or tactics are applicable for the requirement: ``{requirement}``?

    Helpful answer:"""

    COMPARISON_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    The question at the end is about a requirement that this software system must fulfill. 
    The requirement itself is enclosed by backticks. 
    I selected a list of candidate patterns. 
    Your main task is to compare and rank these patterns from best to worst based on the adequacy of each pattern for the requirement.
    Use only patterns from the provided list, but you might discard patterns being unsuitable for the requirement.
    For each ranked pattern include the following data: 
    - pattern name, 
    - a short description, 
    - and concise pros and cons of using the pattern instantiated on the specific requirement; 
    Additional information for contextualizing the selected patterns is provided below. If this information is not present, 
    then return that you don't have sufficient information for the comparison.
    In your answer, return your ranking as a numbered list in textual format.
    In addition, after the ranking, briefly explain how the first pattern in the ranking can be applied to satisfy the requirement.
    If the requirement does not make sense for the context, or is unclear or ambiguous, do not make any comparison and
    just return that you don't have sufficient information.
    Do not try to compare or return irrelevant patterns.

    Patterns: {patterns}

    Context: {system_context}

    Additional information: 
    {pattern_context}

    Question: Can you compare and rank only those patterns above that are applicable for the requirement ``{requirement}``?

    Helpful Answer:"""

    COMPARISON_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed.Additional information can be included.
    The question at the end is about a requirement that this software system must fulfill. 
    The requirement itself is enclosed by backticks. 
    I selected a list of candidate patterns. 
    Your main task is to compare and rank these patterns from best to worst based on the adequacy of each pattern for the requirement.
    Use only patterns from the provided list, but you might discard patterns being unsuitable for the requirement.
    For each ranked pattern include the following data: 
    - pattern name, 
    - a short description, 
    - and concise pros and cons of using the pattern instantiated on the specific requirement; 
    In your answer, return your ranking as a numbered list in textual format.
    In addition, after the ranking, briefly explain how the first pattern in the ranking can be applied to satisfy the requirement.
    If the requirement does not make sense for the context, or is unclear or ambiguous, do not make any comparison and
    just return that you don't have sufficient information.
    Do not try to compare or return irrelevant patterns.

    Patterns: {patterns}

    Context: {system_context}

    Question: Can you compare and rank only those patterns above that are applicable for the requirement ``{requirement}``?

    Helpful Answer:"""

    def __init__(self, sys_store, dk_store, llm, sys_id="DAS-P1-2023", show_prompts=False) -> None:
        self.llm = llm
        self.sys_store = sys_store
        self.dk_store = dk_store
        self.embeddings = SentenceTransformerEmbeddings(model_name=PatternsQACopilot.EMBEDDINGS_MODEL)
        self.target = None

        self.config = {}
        if show_prompts:
            self.config = {"callbacks": [CustomHandler()]}

        self.sys_id = sys_id
        self.reset()
    
    def reset(self):
        result = self.sys_store.get_system(self.sys_id, summary=True)
        self.summary = result['documents'][0] # This is the string with the description
        print(self.summary)
        self.requirement = ""
        self.patterns = []
        self.decision = ""
    
    def configure_retriever(self, collection:str=None, threshold=None):
        
        if threshold is None:
            threshold = PatternsQACopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            selection = PatternSuggestionCopilot.SELECTION_PROMPT_TEMPLATE_ZEROSHOT
            comparison = PatternSuggestionCopilot.COMPARISON_PROMPT_TEMPLATE_ZEROSHOT
        else:
            retriever = self.dk_store.get_retriever(collection=collection, threshold=threshold)
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
            selection = PatternSuggestionCopilot.SELECTION_PROMPT_TEMPLATE_RAG
            comparison = PatternSuggestionCopilot.COMPARISON_PROMPT_TEMPLATE_RAG
        
        self.selection_prompt = ChatPromptTemplate.from_messages([
            ("system", PatternSuggestionCopilot.SYSTEM_PROMPT_TEMPLATE), # system role
            ("human", selection) # human, the user text   
        ])

        self.comparison_prompt = ChatPromptTemplate.from_messages([
            ("system", PatternSuggestionCopilot.SYSTEM_PROMPT_TEMPLATE), # system role
            ("human", comparison) # human, the user text   
        ])

        output_parser1 = SimpleJsonOutputParser()
        self.selection_chain = self.selection_prompt | self.llm | output_parser1

        output_parser2 = StrOutputParser() # TODO: Change the output parser to JSON 
        self.comparison_chain = self.comparison_prompt | self.llm | output_parser2

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    def select_patterns(self, requirement): # TODO: Modificado. Sino, tenía que setear el requirement por fuera
        self.requirement = requirement
        req, patterns = self.fetch_requirement(self.requirement)
        if req is not None:
            self.requirement = req
        
        if len(patterns) == 0:
            retrieved_docs = self.compression_retriever.get_relevant_documents(requirement) 
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) -  selection")

            #if (self.target is None) or (len(retrieved_docs) > 0):
            try:
                patterns = self.selection_chain.invoke({"system_context": self.summary, "requirement": requirement, 
                                                             "pattern_context": PatternSuggestionCopilot._format_docs(retrieved_docs)},
                                                             config=self.config)
            except OutputParserException:
                # print("Error:", e)
                patterns = []
        
        return patterns
    
    def _filter_pattern_queries(self, patterns):               
        # print("Patterns query:", patterns)
        retrieved_docs = []
        mapped_patterns = []
        for p in patterns:
            pquery = "Tell me about the " + p
            pquery_docs = self.compression_retriever.get_relevant_documents(pquery) 
            if len(pquery_docs) > 0:
                # retrieved_docs = retrieved_docs + pquery_docs
                retrieved_docs.append(pquery_docs[0])
                mapped_patterns.append(p)
        
        return retrieved_docs, mapped_patterns

    def compare_patterns_for_requirement(self, patterns): # TODO: Modificado, este tiene que recibir la lista de parámetros, sino, no se puede hacer lo que hablamos ayer
        
        self.patterns = [p+' pattern' if 'pattern' not in p.lower() else p for p in patterns]
        
        retrieved_docs, mapped_patterns = self._filter_pattern_queries(self.patterns)
        print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - comparison")
        if self.target is not None: # It's not zero shot
            print("Mapped patterns:", mapped_patterns)
            self.patterns = mapped_patterns
        # pprint.pprint(retrieved_docs)
        
        comparison = None
        if (len(patterns) > 0) and ((self.target is None) or (len(retrieved_docs) > 0)):
            comparison = self.comparison_chain.invoke({"system_context": self.summary, "requirement": self.requirement, 
                                                       "patterns": self.patterns, 
                                                       "pattern_context": PatternSuggestionCopilot._format_docs(retrieved_docs)},
                                                       config=self.config)

        # TODO: Should we update a new requirement or its associated patterns in the database?

        # TODO: Need to fix this part
        if comparison is not None:
            sp = comparison.split("\n")
            decision = sp[-1]
            comparison = '\n'.join(sp[0:-1])
            return comparison, decision
        else:
            return None, None

    def launch(self):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            requirement = input('>>Requirement: ')
            # give us a way to exit the script
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                # sys.exit()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                # self.requirement = requirement
                self.decision = ""
                self.patterns = self.select_patterns(requirement)
                print(self.requirement)
                print()
                print(">>Candidate patterns:", self.patterns)
                print()

                comparison, decision = self.compare_patterns_for_requirement(self.patterns) 
                
                if comparison is None:
                    comparison = "No comparison available"
                    decision = ""          
                print(">>Answer:", comparison)
                print()
            
                self.decision = decision
                print(">>Decision:", decision)
                print()


class PatternAssessmentCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, then just say so, don't try to make up an answer.
    """

    ANALYSIS_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    There is also a requirement that this software system must fulfill. 
    For that requirement, I made a design decision that is supposed to address the requirement. 
    This decision can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    The question at the end is about my decision, which is enclosed by backticks. 
    I want to better understand the implications of my decision for the context and requirement provided.
    Please think step by step and assess the following aspects in your answer:
    - whether my design decision is appropriate or not. 
    - a list of clarifying questions that expert architects might ask about my decision in the provided context.
    - assumptions and constraints that my decision is based on.
    - consequences of my decision on quality attributes of the system.
    - potential risks and trade-offs implied by the decision.
    - additional follow-up decisions that are necessary to implement my decision.

    Focus on addressing the specific requirement with the decision, rather than on the general use cases for the decision.
    Return your answer in plain text with bullets for the key points of your assessment.
    Carefully check that my design decision is related to the additional information, do not admit dubious or unknown decisions.
    If the decision is dubious or unknown just say so and don't perform any assessment.
    If my decision is not related to the additional information, or the part with addition information is empty,
    or the context is empy, just say that you don't have enough context for the assessment.
    If my decision does not make sense for the context or is not appropriate for the requirement, just state what the problem is.

    Context: {system_context}

    Requirement: {requirement}

    Additional information: 
    {pattern_context}

    Question: What is your assessment of my decision ``{decision}`` for the requirement?

    Helpful Answer:"""

    ANALYSIS_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. There is also a requirement that this software system must fulfill. 
    For that requirement, I made a design decision that is supposed to address the requirement. 
    This decision can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    The question at the end is about my decision, which is enclosed by backticks. 
    I want to better understand the implications of my decision for the context and requirement provided.
    Please think step by step and assess the following aspects in your answer:
    - whether my design decision is appropriate or not. 
    - a list of clarifying questions that expert architects might ask about my decision in the provided context.
    - assumptions and constraints that my decision is based on.
    - consequences of my decision on quality attributes of the system.
    - potential risks and trade-offs implied by the decision.
    - additional follow-up decisions that are necessary to implement my decision.

    Focus on addressing the specific requirement with the decision, rather than on the general use cases for the decision.
    Return your answer in plain text with bullets for the key points of your assessment.
    Carefully check my design decision and do not admit dubious or unknown decisions. 
    If the decision is dubious or unknown just say so and don't perform any assessment.
    If my decision does not make sense for the context or is not appropriate for the requirement, just state what the problem is.

    Context: {system_context}

    Requirement: {requirement}

    Question: What is your assessment of my decision ``{decision}`` for the requirement?

    Helpful Answer:"""

    def __init__(self, sys_store, dk_store, llm, sys_id="DAS-P1-2023", show_prompts=False) -> None:
        self.llm = llm
        self.sys_store = sys_store
        self.dk_store = dk_store
        self.embeddings = SentenceTransformerEmbeddings(model_name=PatternAssessmentCopilot.EMBEDDINGS_MODEL)
        self.target = None

        self.config = {}
        if show_prompts:
            self.config = {"callbacks": [CustomHandler()]}

        self.sys_id = sys_id
        self.reset()
    
    def get_analysis(self):
        return self.analysis
    
    def set_analysis(self, analysis: str):
        self.analysis = analysis
    
    def reset(self):
        result = self.sys_store.get_system(self.sys_id, summary=True)
        self.summary = result['documents'][0] # This is the string with the description
        print(self.summary)
        self.requirement = ""
        self.decision = ""
        self.analysis = ""
    
    def configure_retriever(self, collection:str=None, threshold=None):
        
        if threshold is None:
            threshold = PatternAssessmentCopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            analysis = PatternAssessmentCopilot.ANALYSIS_PROMPT_TEMPLATE_ZEROSHOT
        else:
            retriever = self.dk_store.get_retriever(collection=collection, threshold=threshold)
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
            analysis = PatternAssessmentCopilot.ANALYSIS_PROMPT_TEMPLATE_RAG

        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", PatternAssessmentCopilot.SYSTEM_PROMPT_TEMPLATE), # system role
            ("human", analysis) # human, the user text   
        ])

        output_parser = StrOutputParser()
        self.main_chain = self.analysis_prompt | self.llm | output_parser

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    def assess_decision(self, requirement, decision): 
            self.requirement = requirement
            self.decision = decision
    
            retrieved_docs = self.compression_retriever.get_relevant_documents(decision) 
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - analysis")
    
            analysis = None
            if (self.target is None) or (len(retrieved_docs) > 0):
                analysis = self.main_chain.invoke({"system_context": self.summary, "requirement": requirement, 
                                                "pattern_context": PatternAssessmentCopilot._format_docs(retrieved_docs),
                                                "decision": decision}, config=self.config)
            return analysis
    
    def launch(self):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            self.analysis = ""
            requirement = input('>>Requirement: ')
            # give us a way to exit the script
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                # sys.exit()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                req, patterns = self.fetch_requirement(requirement)
                if req is not None:
                    requirement = req
                print(requirement)
                print(patterns)
                print()
                decision = input('>>Decision: ')
                while decision == "?":
                    decisions_dict = self.get_decisions()
                    pprint.pprint(decisions_dict)
                    reqs = [self.sys_store.get_requirements_for_decision(dd)[0] for dd in decisions_dict.keys()]
                    pprint.pprint(dict(zip(decisions_dict.keys(), reqs)))
                    print()
                    decision = input('>>Decision: ')
                if decision == "exit" or decision == "quit" or decision == "q":
                    print('Bye!')
                    print()
                    # sys.exit()
                    return
                dec, reqs = self.fetch_decision(decision)
                if dec is not None:
                    decision = dec
                print(decision)
                print()
                analysis = self.assess_decision(requirement, decision)
                if analysis is None:
                    analysis = "No assessment available"
                self.analysis = analysis     
                print(">>Answer:", analysis)
                print()
    

class ADRWriterCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to perform the task at the end.
    If you don't know how to perform the task, then just say so, don't try to make up an answer.
    """

    ADR_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and 
    operating enviroment for a software system that needs to be developed. 
    As part of the context, there is also a requirement that this software system must fulfill.
    I made a main design decision that addresses the requirement. 
    For that decision, I might also have an assesment of its pros, cons, and risks, among other aspects.

    Based on all this information, please specify an Architecture Decision Record (ADR) for me.
    The ADR template includes the following sections: 'title', 'motivation', 'decision', 'pros' of the decision, 
    'cons' of the decision, and 'alternatives' considered.
    For writing the sections of the ADR, think step by step and use the following rules:
    1. The 'title' section should be short and descriptive of the purpose of the main design decision.
    2. The 'motivation' section should explain the problem being solved by the decision. Both the system context and 
    the requirement above should be blended in the motivation. Do not use bullets for this section.
    3. The 'decision' section should present the main design decision made and explain how it addresses the requirement.
    Additional information from the assessment about clarifying questions, assumptions and follow-up decisions can
    be also included in this section. Do not use bullets for this section.
    4. The 'pros' and 'cons' sections should list the advantages and disadvantages of the main decision. Information from
    the assessment should be used to populate these sections.
    5. The 'alternatives' section should list any other design decisions that could address the requirement in the system context
    provided, but were not chosen. List up to 3 alternatives and briefly describe each of them. Information from the assessment
    can be used to populate this section.

    Focus on addressing the specific requirement with the decision, rather than on the general use cases for the decision.
    Return your answer in plain text with bullets for the different sections of the ADR.
    Carefully check my design decision and do not admit dubious or unknown decisions. 
    If the decision is dubious or unknown just state the reasons, don't make the ADR up.
    Otherwise, only return the ADR and nothing else.
    
    Context: 
    {system_context}

    Requirement: {requirement}

    Decision: {decision}

    Assessment: 
    {assessment}

    Task: Write an Architecture Decision Record (ADR) based on the context, requirement, decision, and assessment provided.

    Helpful Answer:"""

    def __init__(self, sys_store, dk_store, llm, sys_id="DAS-P1-2023", show_prompts=False) -> None:
        self.llm = llm
        self.sys_store = sys_store
        self.dk_store = dk_store
        self.embeddings = SentenceTransformerEmbeddings(model_name=ADRWriterCopilot.EMBEDDINGS_MODEL)
        self.target = None

        self.config = {}
        if show_prompts:
            self.config = {"callbacks": [CustomHandler()]}

        self.sys_id = sys_id
        self.reset()
    
    def get_analysis(self):
        return self.analysis
    
    def set_analysis(self, analysis: str):
        self.analysis = analysis
    
    def reset(self):
        result = self.sys_store.get_system(self.sys_id, summary=True)
        self.summary = result['documents'][0] # This is the string with the description
        print(self.summary)
        self.requirement = ""
        self.decision = ""
        self.analysis = ""
    
    def configure_retriever(self, collection:str=None, threshold=None):
        
        if threshold is None:
            threshold = ADRWriterCopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            analysis = ADRWriterCopilot.ADR_PROMPT_TEMPLATE_ZEROSHOT
        else:
            retriever = self.dk_store.get_retriever(collection=collection, threshold=threshold)
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
            analysis = ADRWriterCopilot.ADR_PROMPT_TEMPLATE_ZEROSHOT

        self.adr_prompt = ChatPromptTemplate.from_messages([
            ("system", ADRWriterCopilot.SYSTEM_PROMPT_TEMPLATE), # system role
            ("human", analysis) # human, the user text   
        ])

        output_parser = StrOutputParser()
        self.main_chain = self.adr_prompt | self.llm | output_parser

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)

    def write_adr(self, requirement, decision, analysis=""): 
            self.requirement = requirement
            self.decision = decision
    
            # retrieved_docs = self.compression_retriever.get_relevant_documents(decision) 
            # print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - analysis")
    
            adr = None
            if (self.target is None): # or (len(retrieved_docs) > 0):
                adr = self.main_chain.invoke({"system_context": self.summary, "requirement": requirement, 
                                                "assessment": analysis, "decision": decision}, config=self.config)
                if 'sorry' in adr:
                    adr = None
            return adr

    def launch(self):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            self.analysis = ""
            requirement = input('>>Requirement: ')
            # give us a way to exit the script
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                # sys.exit()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                requirement, patterns = self.fetch_requirement(requirement)
                print(requirement)
                print(patterns)
                print()
                decision = input('>>Decision: ')
                while decision == "?":
                    decisions_dict = self.get_decisions()
                    pprint.pprint(decisions_dict)
                    reqs = [self.sys_store.get_requirements_for_decision(dd)[0] for dd in decisions_dict.keys()]
                    pprint.pprint(dict(zip(decisions_dict.keys(), reqs)))
                    print()
                    decision = input('>>Decision: ')
                if decision == "exit" or decision == "quit" or decision == "q":
                    print('Bye!')
                    print()
                    # sys.exit()
                    return
                dec, reqs = self.fetch_decision(decision)
                if dec is not None:
                    decision = dec
                print(decision)
                print()
                adr = self.write_adr(requirement, decision)
                if adr is None:
                    adr = "No assessment available"
                self.analysis = adr     
                print(">>Answer:", adr)
                print()



class PatternConsistencyCheckingCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to perform the task at the end.
    If you don't know how to perform the task, then just say so, don't try to make up an answer.
    """

    CHECKING_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. There is also a requirement that this software system must fulfill. 
    For that requirement, I made a main design decision to address the requirement. 
    My main decision is enclosed by backticks below.
    There are also additional decisions either addressing the same requirement or other requirements of the system.
    
    I want to check the overall consistency of my main decision with the related additional decisions.
    Any of the decisions above can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    
    Report any conflict or inconsistency between the main decision and any of the additional decisions.
    Focus on addressing the specific requirement, rather than on the general use cases for the decisions.
    Return your answer in plain text with bullets for the key points of your analysis.
    Carefully check my main decision and do not admit dubious or unknown decisions. 
    If my main decision is dubious or unknown just say so and don't perform any consistency analysis.
    If my decision does not make sense for the context or is not appropriate for the requirement, just state what the problem is.

    Context: {system_context}

    Requirement: {requirement}
    
    Additional decisions: 
    {additional_decisions}

    Task: Please check the consistency of my main decision ``{decision}`` in the context of the additional decisions

    Helpful Answer:"""

    def __init__(self, sys_store, dk_store, llm, sys_id="DAS-P1-2023", show_prompts=False) -> None:
        self.llm = llm
        self.sys_store = sys_store
        self.dk_store = dk_store
        self.embeddings = SentenceTransformerEmbeddings(model_name=PatternConsistencyCheckingCopilot.EMBEDDINGS_MODEL)
        self.target = None

        self.config = {}
        if show_prompts:
            self.config = {"callbacks": [CustomHandler()]}

        self.sys_id = sys_id
        self.reset()
    
    def get_analysis(self):
        return self.analysis
    
    def set_analysis(self, analysis: str):
        self.analysis = analysis
    
    def reset(self):
        result = self.sys_store.get_system(self.sys_id, summary=True)
        self.summary = result['documents'][0] # This is the string with the description
        print(self.summary)
        self.requirement = ""
        self.decision = ""
        self.analysis = ""
    
    def configure_retriever(self, collection:str=None, threshold=None):
        
        if threshold is None:
            threshold = PatternConsistencyCheckingCopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            analysis = PatternConsistencyCheckingCopilot.CHECKING_PROMPT_TEMPLATE_ZEROSHOT
        else:
            retriever = self.dk_store.get_retriever(collection=collection, threshold=threshold)
            retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
            analysis = PatternConsistencyCheckingCopilot.CHECKING_PROMPT_TEMPLATE_ZEROSHOT

        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", PatternConsistencyCheckingCopilot.SYSTEM_PROMPT_TEMPLATE), # system role
            ("human", analysis) # human, the user text   
        ])

        output_parser = StrOutputParser()
        self.main_chain = self.analysis_prompt | self.llm | output_parser

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    def get_related_decisions(self, requirement, decision):
        # print(decision)
        related_decisions1 = self.sys_store.search_decisions_for_requirement(requirement, semantic_search=True)
        # print("--related by requirement:", list(related_decisions1.keys()))
        related_decisions2 = self.sys_store.search_related_decisions(decision, semantic_search=True)
        # print("--related by decision:", list(related_decisions2.keys()))
        related_decisions1.update(related_decisions2)
        return related_decisions1
    
    def check_decision(self, requirement, decision): 
            self.requirement = requirement
            self.decision = decision

            retrieved_docs = self.compression_retriever.get_relevant_documents(decision) 
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - analysis")
            
            related_decisions = self.get_related_decisions(requirement, decision)
            # Filter out the decision itself
            # related_decisions = { k:v for k,v in related_decisions.items() if v != decision} 
            decision_ids = list(related_decisions.keys())
            related_decisions = list(related_decisions.values())

            analysis = None
            if (self.target is None) or (len(retrieved_docs) > 0):
                print("Related decisions:", decision_ids)
                if len(related_decisions) > 0:
                    formated_decisions = "\n\n".join(doc for doc in related_decisions)
                    analysis = self.main_chain.invoke({"system_context": self.summary, "requirement": requirement, 
                                                "additional_decisions": formated_decisions,
                                                "decision": decision}, config=self.config)
            return analysis
    
    def launch(self):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            self.analysis = ""
            requirement = input('>>Requirement: ')
            # give us a way to exit the script
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                # sys.exit()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                req, patterns = self.fetch_requirement(requirement)
                if req is not None:
                    requirement = req
                print(requirement)
                print(patterns)
                print()
                decision = input('>>Decision: ')
                while decision == "?":
                    decisions_dict = self.get_decisions()
                    pprint.pprint(decisions_dict)
                    reqs = [self.sys_store.get_requirements_for_decision(dd)[0] for dd in decisions_dict.keys()]
                    pprint.pprint(dict(zip(decisions_dict.keys(), reqs)))
                    print()
                    decision = input('>>Decision: ')
                if decision == "exit" or decision == "quit" or decision == "q":
                    print('Bye!')
                    print()
                    # sys.exit()
                    return
                dec, reqs = self.fetch_decision(decision)
                if dec is not None:
                    decision = dec
                print(decision)
                print()
                analysis = self.check_decision(requirement, decision)
                if analysis is None:
                    analysis = "No assessment available"
                self.analysis = analysis     
                print(">>Answer:", analysis)
                print()
