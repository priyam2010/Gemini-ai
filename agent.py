from typing import TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# --- Step 2: Define the Agent State ---
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    Attributes:
        messages: A list of messages that represents the conversation history
                  between the user, the agent, and the tools.
    """
    messages: List[BaseMessage]


# --- Step 3: Define Your Tools ---
@tool
def calculate_roi(investment: float, returns: float) -> str:
    """
    Calculates the Return on Investment (ROI) given the initial investment and total returns.
    Args:
        investment (float): The initial amount of money invested.
        returns (float): The total amount of money returned from the investment.
    Returns:
        A string representing the calculated ROI percentage.
    """
    if investment == 0:
        return "Investment cannot be zero. Please provide a valid investment amount."
    roi = ((returns - investment) / investment) * 100
    return f"The ROI is: {roi:.2f}%"

@tool
def allocate_budget(total_budget: float, platforms: List[str], strategy: str) -> dict:
    """
    Distributes a total budget across specified advertising platforms based on a given strategy.
    Args:
        total_budget (float): The total budget to be distributed.
        platforms (List[str]): A list of advertising platforms (e.g., 'Facebook', 'Instagram', 'YouTube').
        strategy (str): The distribution strategy (e.g., 'focus on ROI', 'equal distribution').
    Returns:
        A dictionary showing the budget allocated to each platform.
    """
    allocation = {}
    if "equal" in strategy.lower() or "distribution" in strategy.lower():
        share = total_budget / len(platforms)
        for platform in platforms:
            allocation[platform] = f"${share:.2f}"
    elif "roi" in strategy.lower():
        if 'YouTube' in platforms:
            allocation['YouTube'] = f"${total_budget * 0.5:.2f}"
        if 'Facebook' in platforms:
            allocation['Facebook'] = f"${total_budget * 0.3:.2f}"
        if 'Instagram' in platforms:
            allocation['Instagram'] = f"${total_budget * 0.2:.2f}"
    else:
        return {"error": "Unknown strategy provided."}

    return allocation

@tool
def get_competitor_data(competitor_name: str) -> dict:
    """
    Fetches simulated campaign data for a given competitor from a dummy API.
    Args:
        competitor_name (str): The name of the competitor.
    Returns:
        A dictionary with simulated campaign metrics (e.g., budget, platforms, ROI).
    """
    dummy_data = {
        "competitor_A": {
            "campaign_name": "Summer_Sales_2025",
            "budget": 250000,
            "platforms": ["Facebook", "Instagram", "Google Ads"],
            "estimated_roi": "150%",
            "top_performer": "Facebook"
        },
        "competitor_B": {
            "campaign_name": "Brand_Awareness_2025",
            "budget": 100000,
            "platforms": ["YouTube", "TikTok"],
            "estimated_roi": "80%",
            "top_performer": "YouTube"
        }
    }
    normalized_name = competitor_name.lower().replace(" ", "_")
    if normalized_name in dummy_data:
        return dummy_data[normalized_name]
    else:
        return {"error": f"No data found for competitor: {competitor_name}"}

tools = [calculate_roi, allocate_budget, get_competitor_data]


# --- Step 4: Create the LangGraph Nodes ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert AI agent specializing in digital marketing and financial calculations. Your task is to assist users with ROI calculations, budget allocation, and competitor analysis. You have access to the following tools: {tool_names}"),
        ("placeholder", "{messages}"),
    ]
).partial(tool_names=", ".join([t.name for t in tools]))

agent_runnable = prompt | llm_with_tools

def agent_node(state):
    messages = state["messages"]

    # Check if the last message is a ToolMessage. If so, create a new
    # HumanMessage to follow it, ensuring the conversation sequence is correct.
    if isinstance(messages[-1], ToolMessage):
        # This is a bit of a hack to satisfy the Gemini API's strict formatting.
        # We are creating a new "human" turn in the conversation.
        messages.append(HumanMessage(content="What should I do next?"))

    result = agent_runnable.invoke({"messages": messages})
    return {"messages": [result]}

def tool_node(state):
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        raise ValueError("Last message from agent has no tool calls.")

    tool_call = last_message.tool_calls[0]
    tool_name = tool_call['name']
    tool_args = tool_call['args']

    for tool_to_run in tools:
        if tool_to_run.name == tool_name:
            result = tool_to_run.invoke(tool_args)
            tool_message = ToolMessage(content=str(result), tool_call_id=tool_call['id'])
            return {"messages": [tool_message]}
    else:
        tool_message = ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call['id'])
        return {"messages": [tool_message]}


# --- Step 5: Assemble and Run the Graph ---
def should_continue(state):
    last_message = state['messages'][-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
workflow.add_edge("tools", "agent")
app = workflow.compile()

print("Agent is ready. Type your requests (e.g., 'What's the ROI on $5000 investment with $7000 returns?').")
print("Type 'exit' to quit.")

while True:
    try:
        user_input = input("> ")
        if user_input.lower() == 'exit':
            break
        
        if not user_input.strip():
            print("Please enter a valid request.")
            continue
        
        input_state = {"messages": [HumanMessage(content=user_input)]}
        
        print("\n--- Agent's thought process ---")
        for chunk in app.stream(input_state):
            print(chunk)
            print("---")
            
        final_state = next(iter(chunk.values()))
        agent_response = final_state['messages'][-1].content
        print(f"\nAgent: {agent_response}")
    except Exception as e:
        print(f"An error occurred: {e}")