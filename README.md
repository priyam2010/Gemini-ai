# AI-Agent-with-External-Tool-Access


An intelligent, multi-tool agent built using **LangGraph** and the **Google Gemini API**. This agent specializes in digital marketing finance, capable of performing complex calculations and data lookups autonomously using a defined set of tools.

---

### ✨ Features

This agent utilizes the **Gemini 1.5 Flash** model and a **LangGraph** state machine to dynamically decide when to converse and when to execute one of its specialized tools.

| Feature | Tool Name | Description |
| :--- | :--- | :--- |
| **ROI Calculation** | `calculate_roi` | Calculates the Return on Investment given the initial investment and total returns. |
| **Budget Allocation** | `allocate_budget` | Distributes a total marketing budget across multiple platforms (Facebook, YouTube, etc.) based on a strategic input. |
| **Competitor Analysis** | `get_competitor_data` | Fetches (simulated) campaign metrics for a specified competitor. |

---

### ⚙️ How to Execute (Installation & Setup)

Follow these steps to set up your environment and run the agent:

#### 1. Prerequisites

* **Python 3.9+**
* **A Gemini API Key:** Obtain one from [Google AI Studio](https://aistudio.google.com/app/apikey).

#### 2. Create the File

Create a file named `agent.py` and paste the Python code from the section below into it.

#### 3. Install Dependencies

Install all required Python libraries:

```bash
pip install langgraph langchain_core langchain-community langchain-google-genai
```
### 4. Run the agent

python agent.py

### graph TD
    A[Start: User Input] --> B(Agent Node: Gemini LLM);
    B -- Model Decides (Conditional Edge) --> C{Router: should_continue};
    C -- YES (Tool Call Detected) --> D(Tools Node: Execute Tool Function);
    D -- Tool Output (ToolMessage) --> B;
    C -- NO (Final Answer) --> E[End: Display Final Response];
