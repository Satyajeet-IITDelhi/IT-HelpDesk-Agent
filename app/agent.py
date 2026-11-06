import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage


class AgentState(TypedDict):
    question: str
    answer: str


llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"]
)


def helpdesk_agent(state: AgentState):
    response = llm.invoke([
        SystemMessage(content="""
        You are an enterprise IT helpdesk assistant.
        Help users with password resets, VPN, laptop access
        and general IT support.
        Never request passwords or confidential credentials.
        If a request needs administrative access, recommend
        contacting the IT support team.
        """),
        ("human", state["question"])
    ])
    return {"answer": response.content}


graph = StateGraph(AgentState)
graph.add_node("helpdesk_agent", helpdesk_agent)
graph.add_edge(START, "helpdesk_agent")
graph.add_edge("helpdesk_agent", END)
agent = graph.compile()


def ask_agent(question: str):
    return agent.invoke({
        "question": question,
        "answer": ""
    })
