import os
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel,Field
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langgraph.graph import StateGraph,START,END
from IPython.display import Image,display
from langgraph.checkpoint.memory import MemorySaver


os.environ['GROQ_API_KEY']=os.getenv(key='GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY']=os.getenv(key='LANGCHAIN_API_KEY')

from langchain_groq import ChatGroq
llm=ChatGroq(model='gemma2-9b-it')
llm.invoke('Hello').content

class Reviewer(BaseModel):
    status:Literal['Approved','Rejected + Feedback']
    feedback:str

class State(TypedDict):
    topic:str 
    source_code:str
    code_review_status:str 
    test_cases:str 
    test_case_review_status:str 
    manager_approval_status:str
    feedback:str
    final_code_version:str
    deployment_status:str

def code_writer(state: State):
    '''Writes code based on the user input'''
    
    topic = state.get('topic')
    feedback = state.get('feedback')
    
    if feedback:
        response = llm.invoke([
            SystemMessage(content=f'You are an expert in writing code in all programming languages. Develop a code based on the user input and take into account the feedback: {feedback}.'),
            HumanMessage(content=f"Here is the user input: {topic}")
        ])
    else:
        response = llm.invoke([
            SystemMessage(content='You are an expert in writing code in all programming languages. Develop a code based on the user input.'),
            HumanMessage(content=f"Here is the user input: {topic}")
        ])

    return {'source_code': response.content}


def code_reviewer(state: State):
    """Reviews the code and provides feedback."""
    
    source_code = state.get('source_code')
    
    reviewer = llm.with_structured_output(Reviewer)
    
    if source_code:
        response = reviewer.invoke([
                SystemMessage(content='You are an expert in reviewing code in all programming languages. Review the following code and provide approval and feedback.'),
                HumanMessage(content=f"Here is the code:\n\n{source_code}")       
               ])

        return {
                 'code_review_status': response.status,
                 'feedback': response.feedback
               }
    
def test_case_generator(state: State):
    """Generates test cases for the given code and user input."""
    
    user_input = state.get('topic')
    source_code = state.get('source_code')    

    if source_code and user_input:
        response = llm.invoke([
                             SystemMessage(content="You are an expert in generating test cases for all programming languages. Generate test cases for the given code and user input."),
                             HumanMessage(content=f"Here is the code:\n\n{source_code}\n\nUser Input: {user_input}")
                              ])
 
    return {'test_cases': response.content}

def test_case_reviewer(state: State):
    """Reviews the generated test cases and verifies them against the provided source code."""
    
    user_input = state.get('topic')
    source_code = state.get('source_code')
    test_cases = state.get('test_cases')    

    reviewer = llm.with_structured_output(Reviewer)
    
    if user_input and source_code and test_cases:
        response = reviewer.invoke([
            SystemMessage(content="You are an expert in reviewing test cases for all programming languages. Review the given code, verify the test cases, and provide your approval along with detailed feedback."),
            HumanMessage(content=f"User Input:\n{user_input}\n\nSource Code:\n{source_code}\n\nTest Cases:\n{test_cases}")
                  ])

        return {
               'test_case_review_status': response.status,
               'feedback': response.feedback
            }


def manager(state:State):
    '''Manager needts to provdie the approval for PROD deployment'''
    return {'manager_approval_status':None,'feedback':None}

def reviewer_status(state:State):
    return state.get('code_review_status')

def test_case_reviewer_status(state:State):
    return state.get('test_case_review_status')

def manager_status(state:State):
    if state['manager_approval_status']=='Approved':
       state.update({'final_code_version':state.get('source_code'),'deployment_status':'LIVE'})
       return state['manager_approval_status']
    else:
       return state['manager_approval_status'] 

# Build the workflow
workflow=StateGraph(State)

# define the nodes
workflow.add_node('Developer',code_writer)
workflow.add_node('Code Reviewer',code_reviewer)
workflow.add_node('Test Case Generator',test_case_generator)
workflow.add_node('Test Case Reviewer',test_case_reviewer)
workflow.add_node('Manager',manager)

# define the edges
workflow.add_edge(START,'Developer')
workflow.add_edge('Developer','Code Reviewer')
workflow.add_conditional_edges('Code Reviewer',reviewer_status,{'Approved':'Test Case Generator','Rejected + Feedback':'Developer'})
workflow.add_edge('Test Case Generator','Test Case Reviewer')
workflow.add_conditional_edges('Test Case Reviewer',test_case_reviewer_status,{'Approved':'Manager','Rejected + Feedback':'Developer'})
workflow.add_edge('Manager',END)
# workflow.add_conditional_edges('Manager',manager_status,{'Approved':END,'Rejected + Feedback':'Developer'})

memory=MemorySaver()

code_agent=workflow.compile(checkpointer=memory)
