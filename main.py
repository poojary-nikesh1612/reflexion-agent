from chains import first_responder, revisor
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, MessagesState, StateGraph
from tool_executer import execute_tools

MAX_ITERATION = 2
RESPONDER = "responder"
REVISOR = "revisor"
EXECUTE_TOOLS = "execute_tools"


def draft_node(state: MessagesState):
    """Draft the initial response."""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def revisor_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState):
    """Determine whether to continue or end based on iteration count."""
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state["messages"])
    num_counts = count_tool_visits

    if num_counts > MAX_ITERATION:
        return END
    return EXECUTE_TOOLS


builder = StateGraph(MessagesState)

builder.add_node(RESPONDER, draft_node)
builder.add_node(REVISOR, draft_node)
builder.add_node(EXECUTE_TOOLS, execute_tools)
builder.set_entry_point(RESPONDER)
builder.add_conditional_edges(
    REVISOR, event_loop, {END: END, EXECUTE_TOOLS: EXECUTE_TOOLS}
)
builder.add_edge(RESPONDER, EXECUTE_TOOLS)
builder.add_edge(EXECUTE_TOOLS, REVISOR)

graph = builder.compile()

# graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    human_message = HumanMessage(
        content="Write about AI-Powered SOC / autonomous soc  problem domain,"
        " list startups that do that and raised capital."
    )

    res = graph.invoke({"messages": [human_message]})
    last_message = res["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
