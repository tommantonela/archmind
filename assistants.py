__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.openai_tools import PydanticToolsParser
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
from langchain_core.pydantic_v1 import BaseModel, Field #, validator

import json
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
    
    def get_requirement(self, as_list=False) -> str:
        # if as_list and isinstance(self.requirement, list):
        #     return [self.requirement]
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
        if all_reqs is not None:
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
                return
        
            ai_msg, retrieved_docs = self.run(query)
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression)")
            # pretty_print_docs(retrieved_docs)
            print()

            print('>>Answer: ', ai_msg.content)
            print()


class PatternAnalysis(BaseModel):
    pattern_name: str = Field(description="name of the pattern")
    description: str = Field(description="description of the pattern")
    pros: Optional[str] = Field(description="list of advantages of using the pattern")
    cons: Optional[str] = Field(description="list of disadvantages of using the pattern")

class PatternRanking(BaseModel):
    ranking: List[PatternAnalysis]
    best_pattern: Optional[str] = Field(description="name of best ranked pattern for the requirements")
    explanation: Optional[str] = Field(description="instantation of the best pattern to satisfy the requirements")
    
class PatternSuggestionCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, then just say return an empty list, don't try to make up an answer.
    """


    SELECTION_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. Remember to focus on the requirements.
    Answer with a plain list of patterns that might be applicable for all the requirements. 
    For each pattern, assess if the pattern is related to the additional information provided. 
    If a pattern is not related to the additional information, do not include the pattern in the list.
    If the part with additional information is empty, just return an empty list.
    If the context is empty, just return an empty list.
    Return the patterns in JSON format. Only include the names of the patterns in the JSON structure, nothing else.
    Double-check if the requirements are clear and unambiguous for the provided context.
    Do not change or alter the original requirements in any way.
    If the requirements do not make sense for the context or are unclear or ambiguous, just return an empty list,
    do not try to provide irrelevant patterns.

    Context: {system_context}

    Additional information: 
    {pattern_context}

    Question: What patterns or tactics are applicable for these requirements: ``{requirements}``?

    Helpful answer:"""


    SELECTION_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. Remember to focus on the requirements.
    Answer with a plain list of patterns that might be applicable for all the requirements. 
    Return the patterns in JSON format. Only include the names of the patterns in the JSON structure, nothing else.
    If the context is empty, just return an empty list,
    Double-check if the requirements are clear and unambiguous for the provided context.
    Do not change or alter the original requirements in any way.
    If the requirements do not make sense for the context or are unclear or ambiguous, just return an empty list,
    do not try to provide irrelevant patterns.

    Context: {system_context}

    Question: What patterns or tactics are applicable for these requirements: ``{requirements}``?

    Helpful answer:"""


    COMPARISON_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. 
    I selected a list of candidate patterns. 
    Your main task is to compare and rank these patterns from best to worst based on the adequacy of each pattern for all the requirements.
    Use only patterns from the provided list, but you might discard patterns being unsuitable for any of the requirements.
    For each ranked pattern include the following data: 
    - pattern name, 
    - a short description, 
    - and concise pros and cons of using the pattern instantiated on the specific requirements; 
    Additional information for contextualizing the selected patterns is provided below. If this information is not present, 
    then return that you don't have sufficient information for the comparison.
    In addition, after the ranking, briefly explain how the best pattern in the ranking can be applied to satisfy all the requirements.
    In your answer, return your ranking as a numbered list and afterwards the explanation for the best pattern, all in {format} format.
    If the requirements do not make sense for the context or are unclear or ambiguous do not make any comparison and
    just return that you don't have sufficient information and explain yourself.
    Do not try to compare or return irrelevant patterns.
    
    Patterns: {patterns}

    Context: {system_context}

    Additional information: 
    {pattern_context}

    Question: Can you compare and rank only those patterns above that are applicable for these requirements ``{requirements}``?

    Helpful Answer:"""


    COMPARISON_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. 
    I selected a list of candidate patterns. 
    Your main task is to compare and rank these patterns from best to worst based on the adequacy of each pattern for all the requirements.
    Use only patterns from the provided list, but you might discard patterns being unsuitable for any of the requirements.
    For each ranked pattern include the following data: 
    - pattern name, 
    - a short description, 
    - and concise pros and cons of using the pattern instantiated on the specific requirements; 
    In addition, after the ranking, briefly explain how the best pattern in the ranking can be applied to satisfy all the requirements.
    In your answer, return your ranking as a list and afterwards the explanation for the best pattern, all in {format} format.
    If the requirement does not make sense for the context, or is unclear or ambiguous, do not make any comparison and
    just return that you don't have sufficient information and explain yourself.
    Do not try to compare or return irrelevant patterns.

    Patterns: {patterns}

    Context: {system_context}

    Question: Can you compare and rank only those patterns above that are applicable for these requirements ``{requirements}``?

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
    
    def configure_retriever(self, collection:str=None, threshold=None, openai_functions=False):
        
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

        if openai_functions:
            output_parser2 = PydanticToolsParser(tools=[PatternRanking, PatternAnalysis])
        else:
            output_parser2 = StrOutputParser() 
        # output_parser2 = JsonOutputParser(pydantic_object=PatternRanking) 
        # output_parser2 = PydanticOutputParser(pydantic_object=PatternRanking) 
        self.comparison_chain = self.comparison_prompt | self.llm | output_parser2

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    def select_patterns(self, requirement): 
        self.requirement = requirement # TODO: This requirement can be a string list
        
        # TODO: This part needs to be improved
        if requirement.startswith("[") and requirement.endswith("]"):
            req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
            requirement = []
            patterns = []
            for r in req_list:
                req, dps = self.fetch_requirement(r.strip())
                if req is not None:
                    requirement.append(req)
                    patterns = patterns + [p for p in dps if p not in patterns]
                else:
                    requirement.append(r.strip())
                self.requirement = '['+','.join(requirement)+']'
        else:   
            req, patterns = self.fetch_requirement(requirement)
            if req is not None:
                requirement = req
                self.requirement = requirement     

        print("Requirement:", self.requirement)
        
        if len(patterns) == 0:
            retrieved_docs = self.compression_retriever.get_relevant_documents(requirement) 
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) -  selection")

            try:
                patterns = self.selection_chain.invoke({"system_context": self.summary, "requirements": self.requirement, 
                                                             "pattern_context": PatternSuggestionCopilot._format_docs(retrieved_docs)},
                                                             config=self.config)
            except OutputParserException:
                patterns = []
        
        return patterns
    
    def _filter_pattern_queries(self, patterns):               
        retrieved_docs = []
        mapped_patterns = []
        for p in patterns:
            pquery = "Tell me about the " + p
            pquery_docs = self.compression_retriever.get_relevant_documents(pquery) 
            if len(pquery_docs) > 0:
                retrieved_docs.append(pquery_docs[0])
                mapped_patterns.append(p)
        
        return retrieved_docs, mapped_patterns

    def compare_patterns_for_requirement(self, patterns, openai_functions=False): 
        self.patterns = [p+' pattern' if 'pattern' not in p.lower() else p for p in patterns]
        
        retrieved_docs, mapped_patterns = self._filter_pattern_queries(self.patterns)
        print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - comparison")
        if self.target is not None: # It's not zero shot
            print("Mapped patterns:", mapped_patterns)
            self.patterns = mapped_patterns

        json_format = 'textual'
        if openai_functions:
            json_format = 'JSON'
        
        comparison = None
        if (len(patterns) > 0) and ((self.target is None) or (len(retrieved_docs) > 0)):
            comparison = self.comparison_chain.invoke({"system_context": self.summary, "requirements": self.requirement, 
                                                       "patterns": self.patterns, "format": json_format,
                                                       "pattern_context": PatternSuggestionCopilot._format_docs(retrieved_docs)},
                                                       config=self.config)
            # print(comparison)

        # TODO: This part needs to reviewed/improved
        if comparison is not None:
            if openai_functions:
                # Warning: Assuming the comparison is a Pydantic object!
                decision = comparison[0].explanation # comparison[0].ranking[0].pattern_name
                if decision is None:
                    decision = comparison[0].ranking[0].pattern_name
                json_list = [x.json() for x in comparison[0].ranking]
                json_list = '['+','.join(json_list)+']'
                return json.loads(json_list), decision
            else:
                decision = None
                if isinstance(comparison, list) and len(comparison) > 0:
                    sp = comparison
                    decision = sp[-1]
                    comparison = "\n".join(sp[0:-1])
                    return comparison, decision
                elif isinstance(comparison, str):
                    sp = comparison.split("\n")
                    decision = sp[-1]
                    comparison = "\n".join(sp[0:-1])
                    return comparison, decision
        
        return None, None
    
    def process_requirements(self, requirement):
        self.decision = ""
        self.patterns = self.select_patterns(requirement)
        print(self.requirement)
        print()
        
        comparison, decision = self.compare_patterns_for_requirement(self.patterns, openai_functions=True) 
        if comparison is None:
            comparison = "No comparison available"
            decision = ""          
        print(">>Answer:")
        pprint.pprint(comparison)
        print()
            
        self.decision = decision
        print(">>Decision:", decision)

        result = dict()
        result['description'] = decision
        result['ranking'] = comparison
        result['requirement'] = requirement
        # result['pattern'] = ""
        
        decision_keys = list(self.sys_store.find_decision(description=decision).keys())
        if len(decision_keys) > 0:
            result['decision'] = decision_keys[0]
        else:
            result['decision']= ""
        return result

    def launch(self, openai_functions=False):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            requirement = input('>>Requirement: ')
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
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

                comparison, decision = self.compare_patterns_for_requirement(self.patterns, openai_functions=openai_functions) 
                
                if comparison is None:
                    comparison = "No comparison available"
                    decision = ""          
                print(">>Answer:")
                if openai_functions:
                    pprint.pprint(comparison)
                else:
                    print(comparison)
                print()
            
                self.decision = decision
                print(">>Decision:", decision)
                print()


class PatternAssessment(BaseModel):
    # decision_name: str = Field(description="name of the decision")
    # description: str = Field(description="description of the decision")
    appropriateness: Optional[str] = Field(description="assessment of the decision as appropriate or not for the requirement plus the reasons for this assessment")
    clarifying_questions: Optional[str] = Field(description="a list of additional questions that expert architects might ask about the decision")
    assumptions_and_constraints: Optional[str] = Field(description="a list of assumptions the decision is based on or constraints imposed by the decision")
    qa_consequences: Optional[str] = Field(description="consequences of the decision on system quality attributes")
    risks_and_tradeoffs: Optional[str] = Field(description="a list of potential risks and tradeoffs implied by the decision")
    followup_decisions: Optional[str] = Field(description="a list of technology and implementation decisions that are necessary to implement the decision")

class PatternAssessmentCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to answer the question at the end.
    If you don't know the answer, then just say so, don't try to make up an answer.
    """


    ANALYSIS_PROMPT_TEMPLATE_RAG = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. Additional information can be included.
    There is also a list of requirements that this software system must fulfill. 
    For these requirements, I made a design decision that is supposed to address all the requirements. 
    This decision can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    The question at the end is about my decision, which is enclosed by backticks. 
    I want to better understand the implications of my decision for the context and requirements provided.
    Please think step by step and assess the following aspects in your answer:
    - whether my design decision is appropriate or not, plus an explanation of your assessment. 
    - a list of clarifying questions that expert architects might ask about my decision in the provided context.
    - assumptions and constraints that my decision is based on.
    - consequences of my decision on quality attributes of the system.
    - potential risks and trade-offs implied by the decision.
    - additional follow-up decisions that are necessary to implement my decision.

    Focus on addressing the specific requirements with the decision, rather than on the general use cases for the decision.
    Return the key points of your assessment in {format} format.
    Carefully check that my design decision is related to the additional information, do not admit dubious or unknown decisions.
    If the decision is dubious or unknown just say so and don't perform any assessment.
    If my decision is not related to the additional information, or the part with addition information is empty,
    or the context is empy, just say that you don't have enough context for the assessment.
    If my decision does not make sense for the context or is not appropriate for the requirements, just state what the problem is.

    Context: {system_context}

    List of requirements: 
    {requirements}

    Additional information: 
    {pattern_context}

    Question: What is your assessment of my decision ``{decision}`` for the requirements above?

    Helpful Answer:"""


    ANALYSIS_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and operating enviroment
    for a software system that needs to be developed. There is also a list of requirements that this software system must fulfill. 
    For these requirements, I made a design decision that is supposed to address all the requirements. 
    This decision can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    The question at the end is about my decision, which is enclosed by backticks. 
    I want to better understand the implications of my decision for the context and requirements provided.
    Please think step by step and assess the following aspects in your answer:
    - whether my design decision is appropriate or not, plus an explanation of your assessment. 
    - a list of clarifying questions that expert architects might ask about my decision in the provided context.
    - assumptions and constraints that my decision is based on.
    - consequences of my decision on quality attributes of the system.
    - potential risks and trade-offs implied by the decision.
    - additional follow-up decisions that are necessary to implement my decision.

    Focus on addressing the specific requirements with the decision, rather than on the general use cases for the decision.
    Return the key points of your assessment in {format} format.
    Carefully check my design decision and do not admit dubious or unknown decisions. 
    If the decision is dubious or unknown just say so and don't perform any assessment.
    If my decision does not make sense for the context or is not appropriate for the requirements, just state what the problem is.

    Context: {system_context}

    List of requirements: 
    {requirements}

    Question: What is your assessment of my decision ``{decision}`` for the requirements above?

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
    
    def configure_retriever(self, collection:str=None, threshold=None, openai_functions=False):
        
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

        if openai_functions:
            output_parser = PydanticToolsParser(tools=[PatternAssessment])
        else:
            output_parser = StrOutputParser() 
        self.main_chain = self.analysis_prompt | self.llm | output_parser

        redundancy_filter = EmbeddingsRedundantFilter(embeddings=self.embeddings)
        pipeline = DocumentCompressorPipeline(transformers=[redundancy_filter])
        self.compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline, base_retriever=retriever)
    
    def assess_decision(self, requirement, decision, openai_functions=False): 
            self.requirement = requirement # TODO: Note that this requirement can be actually a string list
            self.decision = decision
    
            retrieved_docs = self.compression_retriever.get_relevant_documents(decision) 
            print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - analysis")

            formated_requirements = requirement
            if isinstance(requirement, list):
                formated_requirements = "\n\n".join(doc for doc in requirement)
    
            json_format = 'Markdown' #'textual'
            if openai_functions:
                json_format = 'JSON'

            analysis = None
            if (self.target is None) or (len(retrieved_docs) > 0):
                analysis = self.main_chain.invoke({"system_context": self.summary, "requirements": formated_requirements, 
                                                "pattern_context": PatternAssessmentCopilot._format_docs(retrieved_docs),
                                                "decision": decision, "format": json_format}, config=self.config)
            
            if openai_functions:
                temp = json.loads(analysis[0].json())
                return temp
            
            return analysis
    
    def process_requirements_decision(self, requirement, decision):
        if requirement.startswith("[") and requirement.endswith("]"):
            req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
            requirement = []
            for r in req_list:
                req, _ = self.fetch_requirement(r.strip())
                if req is not None:
                    requirement.append(req)
                else:
                    requirement.append(r.strip())
            patterns = None
        else:   
            req, patterns = self.fetch_requirement(requirement)
            if req is not None:
                requirement = req
        print(requirement)
        print(patterns)
        print()

        dec, reqs = self.fetch_decision(decision)
        if dec is not None:
            decision = dec
        print(decision)
        print()
        analysis = self.assess_decision(requirement, decision, openai_functions=True)
        if analysis is None:
            analysis = "No assessment available"
        self.analysis = analysis     
        print(">>Answer:")
        pprint.pprint(analysis)

        return analysis

    def launch(self, openai_functions=False):
        print()
        if self.target is None:
            print("I'm a specialist in: EVERYTHING! (no RAG)")
        else:
            print("I'm a specialist in:", self.target.upper(), "(RAG)")
        # print(openai_functions)
        while True:
            # this prints to the terminal, and waits to accept an input from the user
            self.analysis = ""
            requirement = input('>>Requirement: ')
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                # TODO: This part needs to be improved
                if requirement.startswith("[") and requirement.endswith("]"):
                    req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
                    requirement = []
                    for r in req_list:
                        req, _ = self.fetch_requirement(r.strip())
                        if req is not None:
                            requirement.append(req)
                        else:
                            requirement.append(r.strip())
                    patterns = None
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
                    return
                dec, reqs = self.fetch_decision(decision)
                if dec is not None:
                    decision = dec
                print(decision)
                print()
                analysis = self.assess_decision(requirement, decision, openai_functions=openai_functions)
                if analysis is None:
                    analysis = "No assessment available"
                self.analysis = analysis     
                print(">>Answer:")
                if openai_functions:
                    pprint.pprint(analysis)
                else:
                    print(analysis)
                print()
    

class ADRWriterCopilot(BaseCopilot):

    SYSTEM_PROMPT_TEMPLATE = """You are an experienced software architect that assists a novice developer 
    to design a system. Use the following pieces of context to perform the task at the end.
    If you don't know how to perform the task, then just say so, don't try to make up an answer.
    """


    ADR_PROMPT_TEMPLATE_ZEROSHOT = """The context below describes the main characteristics and 
    operating enviroment for a software system that needs to be developed. 
    As part of the context, there is also a list of requirements that this software system must fulfill.
    I made a main design decision that addresses all the requirements. 
    I might also have an assesment covering several aspects of the decision.
    I might also include an ordered list of design alternatives for the main decision. The main decision is often at the top of this list.

    Based on all this information, please specify an Architecture Decision Record (ADR) for me.
    The ADR template includes the following sections: 'title', 'motivation', 'decision drivers', 'main decision', 
    'alternative decisions' considered, and 'pros' and 'cons' of all the decisions.
    For writing the sections of the ADR, think step by step and use the following rules:
    1. The 'title' section should be short and descriptive of the purpose of the main design decision.
    2. The 'motivation' section should explain the problem being solved by the decision. Both the system context and 
    the requirements above should be blended in the motivation. Do not use bullets for this section.
    3. The 'decision drivers' section should simply list all the given requirements.
    4. The 'main decision' section should present the design decision chosen and explain how it addresses the requirements.
    Additional information from the assessment about clarifying questions, assumptions and follow-up decisions can
    also be included in this section, but integrated into the text. Do not use bullets for this section.
    5. The 'alternatives' section should list any other design decisions that could address the requirements in the system context
    provided, but were not chosen. If an ordered list of design alternatives is included below, please use only those alternatives 
    for this section and describe them briefly. If the ordered list is not included, leave the 'alternatives' section empty.
    6. The 'pros' section should list the advantages of each of the decisions (both main decision and design alternatives). 
    Please organize the advantages according to each of the decisions.
    7. The 'cons' section should list the disadvantages of each of the decisions (both main decision and design alternatives). 
    Please organize the disadvantages according to each of the decisions.
    Information from the assessment below can be used to inform the 'pros' and 'cons' for the main decision.

    Focus on addressing the specific requirements with the decision, rather than on the general use cases for the decision.
    Return your answer in Markdown format considering the different sections of the ADR template.
    Carefully check my design decision and do not admit dubious or unknown decisions. 
    If the decisions are dubious or unknown just state the reasons, don't make the ADR up.
    If everything is Ok, then only return the sections of the ADR and nothing else.
    
    Context: 
    {system_context}

    List of requirements: 
    {requirements}

    Decision: {decision}

    Ranking of alternatives (optional): 
    {ranking}

    Assessment (optional): 
    {assessment}

    Task: Write an Architecture Decision Record based on the context, requirements, decision, and assessment provided.

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
        # print(self.summary)
        self.requirement = ""
        self.decision = ""
        self.analysis = ""
        self.ranking = ""
    
    def configure_retriever(self, collection:str=None, threshold=None):
        
        if threshold is None:
            threshold = ADRWriterCopilot.THRESHOLD
        self.target = collection

        if collection is None:
            retriever = DummyRetriever()
            analysis = ADRWriterCopilot.ADR_PROMPT_TEMPLATE_ZEROSHOT
        else:
            # retriever = self.dk_store.get_retriever(collection=collection, threshold=threshold)
            # retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=self.llm)
            retriever = DummyRetriever()
            self.target = None
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
    
    def process_requirements_decision_analysis(self, requirement, decision, analysis="", ranking=""):
        
        if requirement.startswith("[") and requirement.endswith("]"):
            req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
            requirement = []
            for r in req_list:
                req, _ = self.fetch_requirement(r.strip())
                if req is not None:
                    requirement.append(req)
                else:
                    requirement.append(r.strip())
            patterns = None
        else:   
            req, patterns = self.fetch_requirement(requirement)
            if req is not None:
                requirement = req
        print(requirement)
        print(patterns)
        print()

        dec, reqs = self.fetch_decision(decision)
        if dec is not None:
            decision = dec
        print(decision)
        print()

        print(ranking)
        print()

        return self.write_adr(requirement, decision, analysis, ranking)

    def write_adr(self, requirement, decision, analysis="", ranking=""): 
        self.requirement = requirement # TODO: Note that this requirement could be a list (as a string)
        self.decision = decision
    
        formated_requirements = requirement
        if isinstance(requirement, list):
            formated_requirements = "\n\n".join(doc for doc in requirement)
    
        adr = None
        if (self.target is None): # or (len(retrieved_docs) > 0):
            adr = self.main_chain.invoke({"system_context": self.summary, "requirements": formated_requirements, 
                                            "assessment": analysis, "decision": decision, "ranking": ranking}, 
                                            config=self.config)
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
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                # TODO: This part needs to be improved
                if requirement.startswith("[") and requirement.endswith("]"):
                    req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
                    requirement = []
                    for r in req_list:
                        req, _ = self.fetch_requirement(r.strip())
                        if req is not None:
                            requirement.append(req)
                        else:
                            requirement.append(r.strip())
                    patterns = None
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
                    return
                dec, _ = self.fetch_decision(decision)
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
    for a software system that needs to be developed. There is also a list of requirements that this software system must fulfill. 
    For these requirements, I made a main design decision to intends to address all of them. 
    My main decision is enclosed by backticks below.
    There are also additional decisions either addressing my requirements or other requirements of the system.
    
    I want to check the overall consistency of my main decision with the related additional decisions.
    Any of the decisions above can be expressed as a well-known design pattern, or an architectural style, or an 
    architectural tactic, or a design principle.
    
    Report any conflict or inconsistency between the main decision and any of the additional decisions.
    Focus on addressing the specific requirements, rather than on the general use cases for the decisions.
    Return your answer in plain text with bullets for the key points of your analysis.
    Carefully check my main decision and do not admit dubious or unknown decisions. 
    If my main decision is dubious or unknown just say so and don't perform any consistency analysis.
    If my decision does not make sense for the context or is not appropriate for the requirements, just state what the problem is.

    Context: {system_context}

    List of requirement: 
    {requirements}
    
    Additional decisions: 
    {additional_decisions}

    Task: Please check the consistency of my main decision ``{decision}`` in the context of the requirements and additional decisions

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
        related_decisions1 = self.sys_store.search_decisions_for_requirement(requirement, semantic_search=True, forbidden_decision=decision)
        print("--related by requirement:", list(related_decisions1.keys()))
        related_decisions2 = self.sys_store.search_related_decisions(decision, semantic_search=True)
        print("--related by decision:", list(related_decisions2.keys()))
        related_decisions1.update(related_decisions2)
        return related_decisions1
    
    def check_decision(self, requirement, decision): 
        self.requirement = requirement # TODO: Note that this requirement could be a list (as a string)
        self.decision = decision

        retrieved_docs = self.compression_retriever.get_relevant_documents(decision) 
        print(len(retrieved_docs), "retrieved documents from knowledge base (after compression) - analysis")
            
        related_decisions = self.get_related_decisions(requirement, decision)
        # Filter out the decision itself
        decision_ids = list(related_decisions.keys())
        related_decisions = list(related_decisions.values())
        print("Related decisions:", related_decisions)

        formated_requirements = requirement
        if isinstance(requirement, list):
            formated_requirements = "\n\n".join(doc for doc in requirement)

        analysis = None
        if (self.target is None) or (len(retrieved_docs) > 0):
            print("Related decisions:", decision_ids)
            if len(related_decisions) > 0:
                formated_decisions = "\n\n".join(doc for doc in related_decisions)
                analysis = self.main_chain.invoke({"system_context": self.summary, "requirements": formated_requirements, 
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
            if requirement == "exit" or requirement == "quit" or requirement == "q":
                print('Bye!')
                print()
                return
            elif requirement == "?":
                pprint.pprint(self.get_requirements())
                print()
            else:
                # TODO: This part needs to be improved
                if requirement.startswith("[") and requirement.endswith("]"):
                    req_list = requirement[requirement.find("[")+1:requirement.find("]")].split(',')
                    requirement = []
                    for r in req_list:
                        req, _ = self.fetch_requirement(r.strip())
                        if req is not None:
                            requirement.append(req)
                        else:
                            requirement.append(r.strip())
                    patterns = None
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
                    return
                dec, _ = self.fetch_decision(decision)
                if dec is not None:
                    decision = dec
                print(decision)
                print()
                # related_decisions = self.get_related_decisions(requirement, decision)
                # print(related_decisions)
                analysis = self.check_decision(requirement, decision)
                if analysis is None:
                    analysis = "No assessment available"
                self.analysis = analysis     
                print(">>Answer:", analysis)
                print()
