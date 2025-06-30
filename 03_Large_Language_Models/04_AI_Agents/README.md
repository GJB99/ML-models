# 04_AI_Agents

An AI Agent is a system that can perceive its environment, reason about its goals, and act autonomously to achieve those goals. In the context of Large Language Models, agents are systems that use an LLM as their core reasoning engine.

### How they work:

A common architecture for an LLM-based agent is **ReAct (Reason and Act)**.

1.  **Reason**: The LLM is given a high-level goal. It "thinks" about the problem, breaking it down into smaller, actionable steps. This might involve deciding what information it needs, what tools it should use, or what questions it needs to answer.
2.  **Act**: Based on its reasoning, the agent takes an action. This action often involves using a "tool." A tool can be anything from a simple calculator or a search engine to a complex API or another AI model.
3.  **Observe**: The agent observes the result of its action (e.g., the output from the tool).
4.  **Repeat**: The agent takes this new observation, adds it to its memory, and repeats the "Reason-Act-Observe" loop. It continues this process until it has achieved its original goal.

### Key Components:

-   **LLM Core**: The central "brain" of the agent that performs the reasoning.
-   **Tools**: A set of capabilities the agent can use to interact with the outside world (e.g., web search, code execution, database queries).
-   **Memory**: The agent's ability to remember past actions and observations to inform future decisions. This can be short-term (for the current task) or long-term.
-   **Planning**: The ability to break down a complex goal into a sequence of smaller steps. 