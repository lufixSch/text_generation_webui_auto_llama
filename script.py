import gradio as gr


import extensions.auto_llama.shared as shared

from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool
from extensions.auto_llama.agent import ToolChainAgent, SummaryAgent, ObjectiveAgent
from extensions.auto_llama.llm import OobaboogaLLM
from extensions.auto_llama.config import load_templates, get_active_template
from extensions.auto_llama.ui import tool_chain_agent_tab, tool_tab, summary_agent_tab

from modules import chat, extensions

extension_name = "auto_llama"

params = {
    "display_name": "AutoLLaMa",
    "api_endpoint": "http://localhost:5000",
    "verbose": True,
    "max_iter": 10,
    "do_summary": True,
    "active_templates": {
        "ToolChainAgent": "default",
        "SummaryAgent": "default",
        "ObjectiveAgent": "default",
    },
    "active_tools": ["DuckDuckGo", "Wikipedia"],
}

shared.templates = load_templates()
shared.active_templates = params["active_templates"]

shared.active_tools = set(params["active_tools"])

shared.llm = OobaboogaLLM(params["api_endpoint"])


def create_objective_agent():
    return ObjectiveAgent(
        "ObjectiveAgent",
        get_active_template("ObjectiveAgent"),
        shared.llm,
        verbose=params["verbose"]
    )

def create_tool_chain_agent():
    return ToolChainAgent(
        "ToolChainAgent",
        get_active_template("ToolChainAgent"),
        shared.llm,
        SummaryAgent(
            "SummaryAgent",
            get_active_template("SummaryAgent"),
            shared.llm,
            verbose=params["verbose"],
        ),
        [tool for tool in shared.tools if tool.name in shared.active_tools],
        verbose=params["verbose"],
    )
    
def generate_objective(user_input: str, history: list[tuple[str, str]]):
    chat_messages = ""
    for message, reply in history:
        chat_messages += f"User: {message}\n" if message else ""
        chat_messages += f"Chatbot: {reply}\n" if reply else ""
    
    chat_messages += f"User: {user_input}"
    
    return create_objective_agent().run(chat_messages)[1]

def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.
    """

    # with gr.Box(visible=False, elem_classes="file-saver") as file_saver:
    #    with gr.Row():
    #        template_name_txt = gr.Textbox(value=active_template, label="Template Name")
    #        save_new_btn = gr.Button(value="Save")

    with gr.Accordion("AutoLLaMa", open=False):
        tool_tab()
        tool_chain_agent_tab()
        summary_agent_tab()


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    if user_input[:3] == "/do":
        context_str = "Your reply should be based on this additional context:"

        user_input = user_input.replace("/do", "").lstrip()
        
        objective = generate_objective(user_input, state["history"]["visible"])
        
        res = create_tool_chain_agent().run(
            objective, max_iter=params["max_iter"], do_summary=params["do_summary"]
        )

        old_context = str(state["context"]).strip()
        context_already_included = context_str in old_context

        state["context"] = (
            old_context
            + (
                "\n"
                if context_already_included
                else "\n\nYour reply should be based on this additional context:\n"
            )
            + res[1]
        )

    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result
