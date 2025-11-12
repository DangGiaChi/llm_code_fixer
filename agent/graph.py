from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from agent.tools import python_interpreter

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

llm = ChatOllama(model="mistral:7b-instruct")

tools = [python_interpreter]
llm_with_tools = llm.bind_tools(tools)


AGENT_SYSTEM_PROMPT = """
You are a specialized AI agent designed to fix buggy Python code.
Your goal is to receive a function definition and a buggy implementation, and return a corrected, functional version of the code.

Tool: `python_interpreter`.

Follow this exact workflow:
1.  **Analyze:** Read the provided function prompt and the buggy code.
2.  **Propose Fix:** Write a new, corrected version of the Python function.
3.  **Verify:** To test your fix, you must write unit tests and execute them
    using the `python_interpreter` tool. Your tests should be self-contained
    and print results to stdout.
4.  **Iterate:**
    * If the tests pass and you are confident in the fix, your final
        answer should be ONLY the corrected Python code block.
    * If the tests fail, analyze the error message (stdout/stderr)
        from the `python_interpreter`, refine your hypothesis, and
        go back to step 2 to propose a new fix.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
])

agent_runnable = prompt | llm_with_tools

def run_agent(state: AgentState):
    response = agent_runnable.invoke(state)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return
        
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]
        
        if tool_name == "python_interpreter":
            tool_output = python_interpreter.invoke(tool_input)
            tool_messages.append(ToolMessage(
                content=str(tool_output),
                tool_call_id=tool_call["id"]
            ))
        else:
            tool_messages.append(ToolMessage(
                content=f"Error: Unknown tool {tool_name}",
                tool_call_id=tool_call["id"]
            ))
            
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    return "continue"

def create_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", run_agent)
    workflow.add_node("tools", execute_tools)

    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    
    workflow.add_edge("tools", "agent")

    return workflow.compile()

agent_graph = create_agent_graph()