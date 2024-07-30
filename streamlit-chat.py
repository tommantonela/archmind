from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_community.llms import HuggingFaceEndpoint
# from langchain_groq import ChatGroq

from assistants import PatternsQACopilot, PatternSuggestionCopilot, PatternAssessmentCopilot, ADRWriterCopilot, PatternConsistencyCheckingCopilot
from stores import KnowledgeStore, DesignStore


tt = st.markdown("""
<style>

    .stTabs [aria-selected="true"] {
  		color: #6C60FE;
	}  

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #6C60FE;
        
    }
        
    .stTabs:focus {
        color: #6C60FE;
    }
    
    .stTabs [aria-selected="false"]:hover {
        color: #6C60FE;
    }
     
</style>""", unsafe_allow_html=True)


bb = st.markdown(
            """
            <style>
                button[data-testid="baseButton-primary"]{  
                    background-color: #6C60FE;
                    border-color: #6C60FE;
                    secondaryBackground-color: #BCC2F2;
               }
               button[data-testid="baseButton-primary"]:hover {
                    background-color: #BCC2F2;
                    color:#6C60FE;      
               }
                
            </style>""",
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------

@st.cache_resource()
def create_copilot():
    print('---------------------------- create_copilot')
    chroma_db = "./patterns_chromadb"
    kstore = KnowledgeStore(path=chroma_db)
    
    #llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0, openai_api_key=st.secrets["OPENAI_API_KEY"])    
    llm = HuggingFaceEndpoint(repo_id='mistralai/Mistral-7B-Instruct-v0.2', temperature=0.01, token=st.secrets['HUGGINGFACEHUB_API_TOKEN'])
    #llm = ChatGroq(temperature=0.001, groq_api_key=st.secrets['GROQ_API_KEY'], model_name="mixtral-8x7b-32768")
    
    copilotqa = PatternsQACopilot(store=kstore, llm=llm)
    
    chroma_db2 = "./system_chromadb"
    dstore = DesignStore(path=chroma_db2)
    copilotpat = PatternSuggestionCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
    
    copilotcriticize = PatternAssessmentCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
    
    copilotadr = ADRWriterCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
    
    copilotconsistency = PatternConsistencyCheckingCopilot(sys_store=dstore, dk_store=kstore, llm=llm)
    
    return copilotqa, copilotpat, copilotcriticize, copilotadr, copilotconsistency


def reset_conversation(key,copilot=None):
    st.session_state[key] = []
    if copilot is not None:
        copilot.clear_chat_history()
    return 
   
   
def update_context(key): 
    copilotpat.set_system_summary(st.session_state[key])
    copilotcriticize.set_system_summary(st.session_state[key])
    copilotadr.set_system_summary(st.session_state[key])
    copilotconsistency.set_system_summary(st.session_state[key])
    return
    
    
def test_change(which): 
    print('------------------------ test_change')
    
    if which == 'pattern':
        if 'changed' not in st.session_state:
            st.session_state.changed = -1
        st.session_state.changed += 1
    elif which == 'decision':
        if 'changed_decision' not in st.session_state:
            st.session_state.changed_decision = -1
        st.session_state.changed_decision += 1
    else:
        if 'changed_decision_consistency' not in st.session_state:
            st.session_state.changed_decision_consistency = -1
        st.session_state.changed_decision_consistency += 1
    return
        
        
def click_save_button(button, which):
    if button == 'pattern':
        if which == 'yes':
            st.session_state.byes = True
        else:
            st.session_state.bno = True
    elif button == 'critic':
        if which == 'yes':
            st.session_state.byes_critic = True
        else:
            st.session_state.bno_critic = True
    elif button == 'adr':
        if which == 'yes':
            st.session_state.byes_adr = True
        else:
            st.session_state.bno_adr = True
    else:
        if which == 'yes':
            st.session_state.byes_adr_ass = True
        else:
            st.session_state.bno_adr_ass = True
    return 
        
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

copilotqa, copilotpat, copilotcriticize, copilotadr, copilotconsistency = create_copilot()

if 'qabot' not in st.session_state:
    st.session_state.qabot = []

if 'messages_pat' not in st.session_state:
    st.session_state.messages_pat = []

if 'response_patterns' not in st.session_state:
    st.session_state.response_patterns = []

if 'decisions' not in st.session_state:      
    dd = copilotpat.get_decisions_by_requirement()
    if dd is not None:
       st.session_state.decisions = {k:v[0] for k,v in dd.items() if len(v) > 0}
    else:
       st.session_state.decisions = {}

if 'messages_critic' not in st.session_state:
    st.session_state.messages_critic = []

if 'assessments' not in st.session_state:
    st.session_state.assessments = {}
        
if 'messages_adr' not in st.session_state:
    st.session_state.messages_adr = []
    
if 'messages_consistency' not in st.session_state:
    st.session_state.messages_consistency = []
        
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

# -----------------------------------------------------------------------------

col1, mid, col2 = st.columns([10,1,30])
with col1:
    #st.image(image64)
    st.image('aa.png')
with col2:
    st.write('')
    st.write('')
    st.title('ArchMind')

# ---------------------------------------------------------------

mapping = {'All': 'all', 'Architectural Styles': 'architectural_styles', 
           'Design Patterns':'design_patterns','Microservice Patterns':'microservice_patterns','Zero-shot':None}

option = st.selectbox(
    "Which knowledge source do you want to use?",
           ('All',"Architectural Styles", "Design Patterns", "Microservice Patterns",'Zero-shot'),
           index=0,
           placeholder="",
           )

qa_bot, pattern_suggestion, criticizer, adr_generator, consistency = st.tabs(["Q&A", "Pattern Suggestion", "Pattern Assessment","ADR Generator", 'Consistency Checker'])

# ---------------------------------------------------------------

with consistency:
    
    copilotconsistency.configure_retriever(mapping[option]) 
    
    system_summary = st.text_area(label='System context:',
                                  key='context_consistency',
                                  value=copilotconsistency.get_system_summary(),
                                  help='The context can be edited.',
                                  height=180,
                                  on_change = update_context, args=('context_consistency',))
    
    prompt_consistency = st.chat_input("Enter requirement:",key='req_consistency')
    
    sc = st.container(height=400)
    
    with sc: 
        for message in st.session_state.messages_consistency:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt_consistency:
    
        st.session_state.changed = -1
        st.session_state.changed_decision = -1
        st.session_state.changed_adr = -1
        
        if prompt_consistency == '?':
            mm = "Please, can I get a list of all requirements for which decisions exist?"
            st.session_state.messages_consistency.append({"role": "user", "content": mm}) 
            
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                with sc:
                    with st.chat_message("assistant"):
                        if 'decisions' not in st.session_state or len(st.session_state.decisions) == 0:
                            response = "Sorry, I don't have any requirements with decisions to show you! See me in the previous tabs to analyze requirements, compare patterns and assess decisions! :)"
                            st.write(response)
                            st.session_state.messages_consistency.append({"role": "assistant", "content": response})
                        else:
                            ll = []
                            for k,v in st.session_state.decisions.items():
                                ll.append(f'* **{k}**: {v} ')
                            response_format = "Sure, here is the list of requirements and decisions:\n" + '\n'.join(ll)
                            st.write(response_format)
                            st.session_state.messages_consistency.append({"role": "assistant", "content": response_format})
        else: 
            prompt_consistency = prompt_consistency if not prompt_consistency.replace('.','').isnumeric() else 'RF' + prompt_consistency 
            print(prompt_consistency)
            
            pr, _ = copilotconsistency.fetch_requirement(prompt_consistency)
            pr = pr if pr is not None else prompt_consistency
                                  
            mm = f'Can we discuss the decision for requirement *{pr}*?'
            st.session_state.messages_consistency.append({"role": "user", "content": mm}) 
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                        
            if not prompt_consistency in st.session_state.decisions:
                st.session_state.consistency_requirement = None
                mm = "I'm sorry, I don't have any saved decision for your requirement. See me in the previous tabs to analyze requirements, compare patterns and assess decisions! :)"
                st.session_state.messages_consistency.append({"role": "assistant", "content": mm}) 
                with sc:
                    with st.chat_message("assistant"):
                        st.write(mm)
            else:
                st.session_state.consistency_requirement = (prompt_consistency,pr)
                with sc:
                    with st.chat_message("assistant"):
                        mm = "Sure! See below the decision I have for that requirement. Edit the decision as you see fit and then press ctrl + enter!"
                        st.session_state.messages_consistency.append({"role": "assistant", "content": mm}) 
                        st.write(mm)
                        
            st.session_state.changed_decision_consistency = -1 # para evitar que se retriggeree la búsqueda de abajo
                
    
    if 'consistency_requirement' not in st.session_state or st.session_state.consistency_requirement is None:
        tt = st.text_area(
                key='consistency_decision_input',
                label='Decision to assess:',
                placeholder = 'Once a requirement is analyzed and a decision is saved, it will appear here.',
                help='First, search a requirement for decisions.',
                height = 200,
        )
    else:
        tt = st.text_area(
                key='consistency_decision_input',
                label='Decision to assess:',
                value= st.session_state.decisions[st.session_state.consistency_requirement[0]],
                help='Edit the decision and press ctrl + enter to start the assessment.',
                on_change = test_change,
                args=('consistency',),
                height = 200,
        )
        
        
    if prompt_consistency is None and ('consistency_requirement' not in st.session_state or st.session_state.consistency_requirement is None):
        if tt is not None and len(tt) > 0: 
            with sc:
                with st.chat_message("assistant"):
                    response = "Before assessing the consistency of a decision, please select a requirement :)"
                    st.write(response)
                    st.session_state.messages_consistency.append({"role": "assistant", "content": response})
            st.session_state.changed_decision_consistency = -1
            st.session_state.consistency_requirement = None
    else:
        if 'changed_decision_consistency' in st.session_state and st.session_state.changed_decision_consistency > -1:
            with sc:
                if len(tt) > 0:
                    with st.chat_message("user"): 
                        mm = f'Please, assess the consistency of the following decision: *{tt.strip()}*'
                        st.write(mm)
                        st.session_state.messages_consistency.append({"role": "user", "content": mm})
            
                    with st.chat_message("assistant"): 
                        # if 'context_consistency' in st.session_state and st.session_state.context_consistency != '':
                            # copilotadr.set_system_summary(st.session_state.context_consistency)
                       
                        prev_decisions = copilotconsistency.get_related_decisions(st.session_state.consistency_requirement[0], tt)
                        print(prev_decisions)
                        
                        if len(prev_decisions) != 0:

                            mm = "For your requirement and decision, I've retrieved the following decisions to include in the consistency check:\n* " + '\n* '.join(prev_decisions.values())
                            st.write(mm)
                            st.session_state.messages_consistency.append({"role": "assistant", "content": mm})
                           
                            analysis = copilotconsistency.check_decision(st.session_state.consistency_requirement[0], tt)
                            
                            if analysis is None:
                                mm = "I'm sorry, I cannot provide any assessment of the consistency of your decision."
                                st.write(mm)
                                st.session_state.messages_consistency.append({"role": "assistant", "content": mm})
                            else:
                                mm = 'Here is my consistency assessment for your requirement and decision:\n ' + analysis
                                st.write(mm)
                                st.session_state.messages_consistency.append({"role": "assistant", "content": mm})
                        else:
                            mm = "I'm sorry, I didn't find any related decision for your requirement. To provide an assessment there must be at least one previous related decision."
                            st.write(mm)
                            st.session_state.messages_consistency.append({"role": "assistant", "content": mm})
                            
    st.button("Reset conversation", key='reset_consistency', type="primary", use_container_width=True, on_click=reset_conversation, args=('messages_consistency',)) 

# ---------------------------------------------------------

with adr_generator:
    
    copilotadr.configure_retriever(mapping[option]) 
    
    system_summary = st.text_area(label='System context:',
                                  key='context_adr',
                                  value=copilotadr.get_system_summary(),
                                  help='The context can be edited.',
                                  height=180,
                                  on_change = update_context, args=('context_adr',))
     

    prompt_adr = st.chat_input("Enter requirement:",key='req_adr')
    
    sc = st.container(height=400)
    
    with sc: 
        for message in st.session_state.messages_adr:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if ('byes_adr_ass' in st.session_state and st.session_state.byes_adr_ass) or ('bno_adr_ass' in st.session_state and st.session_state.bno_adr_ass):
        
        with sc:
           with st.chat_message("assistant"):
               if 'byes_adr_ass' in st.session_state and st.session_state.byes_adr_ass:
                    st.session_state.include_assessment = True
                    mm = "Sure! See below the decision and the assessment I have for that requirement. Edit the decision and assessment as you see fit and then click the button!"
                    st.session_state.messages_adr.append({"role": "assistant", "content": mm}) 
                    st.write(mm)
                    st.session_state.byes_adr_ass = False
               else:
                    st.session_state.include_assessment = False
                    mm = "Sure! See below the decision I have for that requirement. Edit the decision as you see fit and then click the button!"
                    st.session_state.messages_adr.append({"role": "assistant", "content": mm}) 
                    st.write(mm)
                    st.session_state.bno_adr_ass = False
                    
               st.session_state.changed_adr = -1 # para evitar que se retriggeree la búsqueda de abajo
    
    elif ('byes_adr' in st.session_state and st.session_state.byes_adr) or ('bno_adr' in st.session_state and st.session_state.bno_adr):
        with sc:
           with st.chat_message("assistant"):
               if 'byes_adr' in st.session_state and st.session_state.byes_adr:
                   response = f"Sure! I've just downloaded the ADR for requirement: *{st.session_state.adr_requirement[1]}*"
                   st.markdown(response)
                   st.session_state.messages_adr.append({"role": "assistant", "content": response})               
                   st.session_state.byes_adr = False
               else:
                   response = "No problem! I've not downloaded the ADR."
                   st.markdown(response)
                   st.session_state.messages_critic.append({"role": "assistant", "content": response})
                   st.session_state.bno_adr = False

               st.session_state.changed_adr = -1 # para evitar que se retriggeree la búsqueda de abajo

    elif prompt_adr:
    
        st.session_state.changed = -1
        st.session_state.changed_decision = -1
    
        if prompt_adr == '?':
            mm = "Please, can I get a list of all requirements for which decisions exist?"
            st.session_state.messages_adr.append({"role": "user", "content": mm}) 
            
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                with sc:
                    with st.chat_message("assistant"):
                        if 'decisions' not in st.session_state or len(st.session_state.decisions) == 0:
                            response = "Sorry, I don't have any requirements with decisions to show you! See me in the previous tabs to analyze requirements, compare patterns and assess decisions! :)"
                            st.write(response)
                            st.session_state.messages_adr.append({"role": "assistant", "content": response})
                        else:
                            ll = []
                            for k,v in st.session_state.decisions.items():
                                ll.append(f'* **{k}**: {v} ')
                            response_format = "Sure, here is the list of requirements and decisions:\n" + '\n'.join(ll)
                            st.write(response_format)
                            st.session_state.messages_adr.append({"role": "assistant", "content": response_format})
        else: 
            prompt_adr = prompt_adr if not prompt_adr.replace('.','').isnumeric() else 'RF' + prompt_adr 
            print(prompt_adr)

            pr, _ = copilotpat.fetch_requirement(prompt_adr)
            pr = pr if pr is not None else prompt_adr
            
            st.session_state.adr_requirement = (prompt_adr,pr)
            
            mm = f'Can your generate an ADR for requirement *{pr}*?'
            st.session_state.messages_adr.append({"role": "user", "content": mm}) 
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                        
            if not prompt_adr in st.session_state.decisions:
                st.session_state.adr_requirement = None
                mm = "I'm sorry, I don't have any saved decision for your requirement. See me in the previous tabs to analyze requirements, compare patterns and assess decisions! :)"
                st.session_state.messages_adr.append({"role": "assistant", "content": mm}) 
                with sc:
                    with st.chat_message("assistant"):
                        st.write(mm)
            else: 
                st.session_state.adr_requirement = (prompt_adr,pr)
                with sc:
                    with st.chat_message("assistant"):
                        if not st.session_state.adr_requirement[0] in st.session_state.assessments: 
                            st.session_state.include_assessment = False
                            mm = "Sure! See below the decision I have for that requirement. Edit the decision as you see fit and then click the button!"
                            st.session_state.messages_adr.append({"role": "assistant", "content": mm}) 
                            st.write(mm)
                        else:
                        
                            mm = 'Sure! For that requirement I also found an assessment of the decision. Do you want to also use the assessment to generate the ADR?'
                            st.session_state.messages_adr.append({"role": "assistant", "content": mm}) 
                            st.write(mm)
                            
                            byes_adr_ass, bno_adr_ass, _ = st.columns([0.1,0.1,0.8])
                            with byes_adr_ass:
                                st.button('Yes',on_click=click_save_button, args=('adr_assessment','yes',))
                            with bno_adr_ass:
                                st.button('No', on_click=click_save_button, args=('adr_assessment','no',))
                        
            st.session_state.changed_adr = -1 # para evitar que se retriggeree la búsqueda de abajo
            
    with st.form("form_adr"):
        if 'adr_requirement' not in st.session_state or st.session_state.adr_requirement is None: 
            tt = st.text_area(
                    key='input_adr_decision',
                    label='Decision to create the ADR for:',
                    placeholder = 'Once a requirement is analyzed and a decision is saved, it will appear here.',
                    help='First, search a requirement for decisions.',
                    height = 200,
            )
        else:
            tt = st.text_area(
                    key='input_adr_decision',
                    label='Decision to create the ADR for:',
                    value= st.session_state.decisions[st.session_state.adr_requirement[0]],
                    help='Edit the decision as you see fit',
                    args=('decision',),
                    height = 200,
            )
            
        if 'include_assessment' in st.session_state and st.session_state.include_assessment: 
            tt = st.text_area(
                    key='input_adr_assessment',
                    label='Assessment of decision:',
                    value = st.session_state.assessments[st.session_state.adr_requirement[0]],
                    help='First, search a requirement for decisions and assessments.',
                    height = 200,
            )
              
        button_adr = st.form_submit_button("Generate ADR",type="primary", use_container_width=True)
        
    if button_adr: 
        
        if not 'adr_requirement' in st.session_state or st.session_state.adr_requirement is None:
            response = "Before creating an ADR, please select a requirement :)"
            with sc:
                with st.chat_message("assistant"):
                    st.write(response)
            st.session_state.messages_adr.append({"role": "assistant", "content": response})
        else:
        
            mm = f'Please, generate the ADR for the requirement *{st.session_state.adr_requirement[1]}* and its decision'
            with sc:
                with st.chat_message("user"):
                    st.write(mm)
            st.session_state.messages_adr.append({"role": "user", "content": mm})
            
            if 'input_adr_decision' in st.session_state:
                decision = st.session_state.input_adr_decision
            
            if 'input_adr_assessment' in st.session_state:
                assessment = st.session_state.input_adr_assessment
            else:
                assessment = ''
                
            if decision is None or decision == '':
                mm = f'The decision is empty! Please select a different requirement or write a decision.'
                with sc:
                    with st.chat_message("assistant"):
                        st.write(mm)
                    st.session_state.messages_adr.append({"role": "assistant", "content": mm})
            else:
                # if 'context_adr' in st.session_state and st.session_state.context_adr != '':
                    # copilotadr.set_system_summary(st.session_state.context_adr)
                    
                adr = copilotadr.write_adr(st.session_state.adr_requirement[0], decision, assessment)
                if adr is None:
                    mm = "I'm sorry, I can't generate an ADR for the selected requirement and decision."
                    with sc:
                        with st.chat_message("assistant"):
                            st.write(mm)
                    st.session_state.messages_adr.append({"role": "assistant", "content": mm})
                else:
                    mm = f"Sure! Here is the generated ADR:\n *{adr}*"
                    with sc:
                        with st.chat_message("assistant"):
                            st.write(mm)
                    st.session_state.messages_adr.append({"role": "assistant", "content": mm})
                    
                    st.session_state.last_adr = adr
                    with sc:
                        with st.chat_message("assistant"):
                            mm = 'Do you want to download the generated ADR?'
                            st.write(mm)
                            
                            st.session_state.messages_adr.append({"role": "assistant", "content": mm})
                                    
                            byes_adr, bno_adr, _ = st.columns([0.1,0.1,0.8])
                            with byes_adr:
                                st.download_button(label="Yes",
                                                   data= st.session_state.last_adr,
                                                   file_name= f"adr_{st.session_state.adr_requirement[0]}.txt",
                                                   mime='application/txt',
                                                   on_click=click_save_button, args=('adr','yes',))
                            with bno_adr:
                                st.button('No', on_click=click_save_button, args=('adr','no',))

    st.button("Reset conversation", key='reset_adr', type="primary", use_container_width=True, on_click=reset_conversation, args=('messages_adr',)) 
    
# --------------------------------------------------------------------------------------            
            
with criticizer:

    copilotcriticize.configure_retriever(mapping[option]) 
    
    system_summary = st.text_area(label='System context:',
                                  key='context_critic',
                                  value=copilotcriticize.get_system_summary(),
                                  help='The context can be edited.',
                                  height=180,
                                  on_change = update_context, args=('context_critic',))
    
    prompt_critic = st.chat_input("Enter requirement:",key='req_critic')
    
    sc = st.container(height=400)
    
    with sc: 
        for message in st.session_state.messages_critic:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if ('byes_critic' in st.session_state and st.session_state.byes_critic) or ('bno_critic' in st.session_state and st.session_state.bno_critic):
        with sc:
           with st.chat_message("assistant"):
               if 'byes_critic' in st.session_state and st.session_state.byes_critic:
                   response = f"Sure! I've just saved the assessment of the decision for requirement: *{st.session_state.critic_requirement[1]}*"
                   st.markdown(response)
                   st.session_state.messages_critic.append({"role": "assistant", "content": response})

                   st.session_state.assessments[st.session_state.critic_requirement[0]] = st.session_state.last_assessment

                   response = "If you want me to generate the corresponding ADR, just see me at the next tab! :)"
                   st.markdown(response)
                   st.session_state.messages_critic.append({"role": "assistant", "content": response})
                   st.session_state.byes_critic = False
               else:
                   response = "No problem! I've not saved the assessment."
                   st.markdown(response)
                   st.session_state.messages_critic.append({"role": "assistant", "content": response})
                   st.session_state.bno_critic = False

               st.session_state.changed_decision = -1 # para evitar que se retriggeree la búsqueda de abajo

    elif prompt_critic:
    
        st.session_state.changed = -1
        
        if prompt_critic == '?':
            mm = "Please, can I get a list of all requirements for which decisions exist?"
            st.session_state.messages_critic.append({"role": "user", "content": mm}) 
            
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                with sc:
                    with st.chat_message("assistant"):
                        if 'decisions' not in st.session_state or len(st.session_state.decisions) == 0:
                            response = "Sorry, I don't have any requirements with decisions to show you! See me in the previous tab to analyze requirements and compare patterns! :)"
                            st.write(response)
                            st.session_state.messages_critic.append({"role": "assistant", "content": response})
                        else:
                            ll = []
                            for k,v in st.session_state.decisions.items():
                                ll.append(f'* **{k}**: {v} ')
                            response_format = "Sure, here is the list of requirements and decisions:\n" + '\n'.join(ll)
                            st.write(response_format)
                            st.session_state.messages_critic.append({"role": "assistant", "content": response_format})
        else: 
            prompt_critic = prompt_critic if not prompt_critic.replace('.','').isnumeric() else 'RF' + prompt_critic 
            print(prompt_critic)
            
            pr, _ = copilotpat.fetch_requirement(prompt_critic)
            pr = pr if pr is not None else prompt_critic
                       
            st.session_state.critic_requirement = (prompt_critic,pr)
                       
            mm = f'Can we discuss the decision for requirement *{pr}*?'
            st.session_state.messages_critic.append({"role": "user", "content": mm}) 
            with sc:
                with st.chat_message("user"):        
                    st.markdown(mm)
                        
            if not prompt_critic in st.session_state.decisions:
                st.session_state.critic_requirement = None
                mm = "I'm sorry, I don't have any saved decision for your requirement. See me in the previous tab to analyze requirements and compare patterns! :)"
                st.session_state.messages_critic.append({"role": "assistant", "content": mm}) 
                with sc:
                    with st.chat_message("assistant"):
                        st.write(mm)
            else:
                with sc:
                    with st.chat_message("assistant"):
                        mm = "Sure! See below the decision I have for that requirement. Edit the decision as you see fit and then press ctrl + enter!"
                        st.session_state.messages_critic.append({"role": "assistant", "content": mm}) 
                        st.write(mm)
                        
            st.session_state.changed_decision = -1 # para evitar que se retriggeree la búsqueda de abajo
            
       
    if 'critic_requirement' not in st.session_state or st.session_state.critic_requirement is None:
        tt = st.text_area(
                key='decision_input',
                label='Decision to assess:',
                placeholder = 'Once a requirement is analyzed and a decision is saved, it will appear here.',
                help='First, search a requirement for decisions.',
                height = 200,
        )
    else:
        tt = st.text_area(
                key='decision_input',
                label='Decision to assess:',
                value= st.session_state.decisions[st.session_state.critic_requirement[0]],
                help='Edit the decision and press ctrl + enter to criticize.',
                on_change = test_change,
                args=('decision',),
                height = 200,
        )
        
        
    if prompt_critic is None and (not 'critic_requirement' in st.session_state or st.session_state.critic_requirement is None)  :
            if tt is not None and len(tt) > 0: 
                with sc:
                    with st.chat_message("assistant"):
                        response = "Before assessing a decision, please select a requirement :)"
                        st.write(response)
                        st.session_state.messages_critic.append({"role": "assistant", "content": response})
                        st.session_state.changed_decision = -1
                        
    else:
        if 'changed_decision' in st.session_state and st.session_state.changed_decision > -1:
            with sc:
                if len(tt) > 0:
                    with st.chat_message("user"): 
                        mm = f'Please, assess the following decision: {tt}'
                        st.write(mm)
                        st.session_state.messages_critic.append({"role": "user", "content": mm})
            
                    with st.chat_message("assistant"): 
                        # if 'context_critic' in st.session_state and st.session_state.context_critic != '':
                            # copilotcriticize.set_system_summary(st.session_state['context_critic'])
                        
                        analysis = copilotcriticize.assess_decision(st.session_state.critic_requirement, tt)
                        if analysis is None:
                            mm = "I'm sorry, I cannot provide any assessment for your decision."
                            st.write(mm)
                            st.session_state.messages_critic.append({"role": "assistant", "content": mm})
                        else:
                            st.write(analysis)
                            st.session_state.messages_critic.append({"role": "assistant", "content": analysis})
                            mm = 'Do you want me to save the assessment for the decision?'
                            st.write(mm)
                            st.session_state.messages_critic.append({"role": "assistant", "content": mm})
                            
                            st.session_state.last_assessment = analysis

                            byes_critic, bno_critic, _ = st.columns([0.1,0.1,0.8])
                            with byes_critic:
                                st.button('Yes',on_click=click_save_button, args=('critic','yes',))
                            with bno_critic:
                                st.button('No', on_click=click_save_button, args=('critic','no',))
                    
                    
    st.button("Reset conversation", key='reset_critic', type="primary", use_container_width=True, on_click=reset_conversation, args=('messages_critic',))                     
# ------------------------------------------------------------------------

with pattern_suggestion:
    
    copilotpat.configure_retriever(mapping[option])
    
    system_summary = st.text_area(label='System context:',
                                  key= 'context_pattern',
                                  value=copilotpat.get_system_summary(),
                                  help='The context can be edited.',
                                  height=180,
                                  on_change = update_context, args=('context_pattern',))
        
    prompt = st.chat_input("Enter requirement:")
    
    sc = st.container(height=400)
    
    with sc: 
        for message in st.session_state.messages_pat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
    if ('byes' in st.session_state and st.session_state.byes) or ('bno' in st.session_state and st.session_state.bno):
        with sc:
            with st.chat_message("assistant"):
                if 'byes' in st.session_state and st.session_state.byes:
                    response = f"Sure! I just saved the decision for requirement: *{st.session_state.pat_requirement[1]}*"
                    st.markdown(response)
                    st.session_state.messages_pat.append({"role": "assistant", "content": response})
                    
                    st.session_state.decisions[st.session_state.pat_requirement[0]] = st.session_state.last_decision
                    
                    response = "If you want me to assess the decision, just see me at the next tab! :)"
                    st.markdown(response)
                    st.session_state.messages_pat.append({"role": "assistant", "content": response})
                    st.session_state.byes = False
                    st.session_state.byes = False
                else:
                    response = "No problem! I didn't save the decision."
                    st.markdown(response)
                    st.session_state.messages_pat.append({"role": "assistant", "content": response})
                    st.session_state.bno = False 
                
                st.session_state.changed = -1 # para evitar que se retriggeree la búsqueda de abajo

    elif prompt:
        if prompt == '?':
            st.session_state.messages_pat.append({"role": "user", "content": "Please, can I get a list of all requirements?"}) 
            with sc:
                with st.chat_message("user"):        
                    st.markdown("Please, can I get a list of all requirements?")
                with sc:
                    with st.chat_message("assistant"):
                        response_format = copilotpat.get_requirements()
                        ll = []
                        for k,v in response_format.items():
                            ll.append(f'* **{k}**: {v} ')
                        response_format = 'Sure, here is the list of requirements:\n' + '\n'.join(ll)
                        st.write(response_format)
                        st.session_state.messages_pat.append({"role": "assistant", "content": response_format})
        else:
            prompt = prompt if not prompt.replace('.','').isnumeric() else 'RF' + prompt 
            print(prompt)
            
            pr, _ = copilotpat.fetch_requirement(prompt)
            pr = pr if pr is not None else prompt
            
            st.session_state.pat_requirement = (prompt,pr)
            
            st.session_state.messages_pat.clear() 
            
            st.session_state.messages_pat.append({"role": "user", "content": pr}) 
            with sc:
                with st.chat_message("user"):        
                    st.markdown(pr)
            with sc:
                with st.chat_message("assistant"):
                    # if 'context_pattern' in st.session_state and st.session_state.context_pattern != '':
                        # copilotpat.set_system_summary(st.session_state['context_pattern'])
                        
                    st.session_state.response_patterns = copilotpat.select_patterns(prompt)
                    if 'changed' in st.session_state:
                        st.session_state.changed = -1
                    if len(st.session_state.response_patterns) > 0:
                        response_format = "For your requirement, I have the following patterns: \n* " + '\n* '.join(st.session_state.response_patterns)
                        response_format += '\n\n Please, edit the list of patterns to be compared and ranked and press ctrl + enter.'
                    else:
                        response_format = "I'm sorry, I couldn't find any pattern for your requirement. Please, try again."
                        
                    st.write(response_format)
                    st.session_state.messages_pat.append({"role": "assistant", "content": response_format})
    
       
    print(st.session_state.response_patterns)
    if st.session_state.response_patterns is None or len(st.session_state.response_patterns) == 0:
        tt = st.text_area(
                key='pattern_input',
                label='Patterns to select:',
                placeholder = 'Once a requirement is analyzed and patterns are found, they will appear here for analysis.',
                help='First, search a requirement for patterns.',
                height = 100
        )
    else:
        tt = st.text_area(
                key='pattern_input',
                label='Patterns to select:',
                value=', '.join(st.session_state.response_patterns),
                help='Edit the list of patterns and press ctrl + enter to search.',
                on_change = test_change,
                args=('pattern',),
                height = 100
        )
       
    if st.session_state.response_patterns is None or len(st.session_state.response_patterns) == 0:
        if tt is not None and len(tt) > 0:
            with sc:
                with st.chat_message("assistant"):
                    response = "Before searching for patterns, first ask me about a requirement :)"
                    st.write(response)
                    st.session_state.messages_pat.append({"role": "assistant", "content": response})
                        
    else: 
        if 'changed' in st.session_state and st.session_state.changed > -1:
            with sc:
                if len(tt) > 0:
                    with st.chat_message("user"): 
                        st.write('Please, compare and rank the following patterns: '+tt)
                        st.session_state.messages_pat.append({"role": "user", "content": 'Please, compare and rank the following patterns: '+tt})
            
                with st.chat_message("assistant"): 
                    tt = tt.split(',')
                    tt = [x for x in tt if len(x) > 0 and x != '']
                    if len(tt) > 0: 
                    
                        if 'context_pattern' in st.session_state and st.session_state.context_pattern != '':
                            copilotpat.set_system_summary(st.session_state['context_pattern'])
                    
                        comparison, decision = copilotpat.compare_patterns_for_requirement(tt)
                                        
                        if comparison is None:
                            response = "I'm sorry, I cannot rank the patters you provided. Please, try again."
                            st.write(response)
                            st.session_state.messages_pat.append({"role": "assistant", "content": response})
                        else:
                            st.write(comparison)
                            st.session_state.messages_pat.append({"role": "assistant", "content": comparison})
                            decision = decision.strip()
                            if decision.startswith('- Application:'):
                                decision = decision.replace('- Application:','') 
                            elif decision.startswith('Application:'):
                                decision = decision.replace('Application:','') 
                                
                            st.write(f'*{decision}*')
                            st.session_state.messages_pat.append({"role": "assistant", "content": f'*{decision}*'})
                            st.session_state.last_decision = decision # TODO: FIX DECISION IDENTIFICATION
                            
                            st.write('Do you want me to save the decision for the requirement?')
                            st.session_state.messages_pat.append({"role": "assistant", "content": 'Do you want me to save the decision for the requirement?'})
                            
                            byes, bno, _ = st.columns([0.1,0.1,0.8])
                            with byes: 
                                st.button('Yes',on_click=click_save_button, args=('pattern','yes',))
                            with bno:
                                st.button('No', on_click=click_save_button, args=('pattern', 'no',))
                            
                    else:
                        response = "Please, provide at least one pattern to assess."
                        st.write(response)
                        st.session_state.messages_pat.append({"role": "assistant", "content": response})


    st.button("Reset conversation", key='reset_pattern', type="primary", use_container_width=True, on_click=reset_conversation, args=('messages_pat',))   

# -----------------------------------------------------------------------------


with qa_bot:
               
    copilotqa.configure_retriever(mapping[option]) 
        
    sc = st.container(height=400)

    with sc:
        for message in st.session_state.qabot:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about patterns!"):
        st.session_state.qabot.append({"role": "user", "content": prompt})
        with sc:
            with st.chat_message("user"):
                st.markdown(prompt)
        with sc:
            with st.chat_message("assistant"):
                response = copilotqa.run(prompt)
                response = response[0] if isinstance(response[0],str) else response[0].content
                st.write(response)
        st.session_state.qabot.append({"role": "assistant", "content": response})
    
    

    
    st.button("Reset conversation", key='reset_qa', type="primary", use_container_width=True, on_click=reset_conversation, args=('qabot',copilotqa)) 
