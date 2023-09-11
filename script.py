from extensions.auto_llama.tool import WikipediaTool
from extensions.auto_llama.agent import BaseAgent, PromptTemplate
from extensions.auto_llama.llm import OobaboogaLLM

from modules import chat

params = {
    "display_name": "AutoLLaMa",
    "api_endpoint": "http://localhost:5000"
}

template = PromptTemplate(
    "Action",
    "Action Input",
    "Observation",
    "Thought",
    "Final Answer",
    """###SYSTEM:Answer the following questions as best you can, You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tools_keywords}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input or the final conclusion to your thoughts


Begin!

###USER: {objective}
###ASSISTANT: {agent_scratchpad}"""
)

tools = [
    WikipediaTool()
]

llm = OobaboogaLLM(params['api_endpoint'], stopping_strings=[f'\n{template.observation_keyword}'])

agent = BaseAgent('AutoLLaMa', template, llm, tools, verbose=True)


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    
    if user_input[:3] == "/do":
        context_str = "Your reply should be based on this additional context:"
        
        objective = user_input.replace("/do", "").lstrip()
        res = agent.run(objective)
        
        print(f"AutoLLaMa Result: {res}")
        
        old_context = str(state['context']).strip()
        context_already_included = context_str in old_context
        
        state['context'] = old_context + ('\n' if context_already_included else '\nYour reply should be based on this additional context:\n') + res[1]
    
    
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result