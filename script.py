import gradio as gr


import extensions.auto_llama.shared as shared

from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool
from extensions.auto_llama.agent import BaseAgent, PromptTemplate
from extensions.auto_llama.llm import OobaboogaLLM
from extensions.auto_llama.config import load_templates
from extensions.auto_llama.ui import template_tab, tool_tab

from modules import chat, extensions

extension_name = "auto_llama"

params = {
    "display_name": "AutoLLaMa",
    "api_endpoint": "http://localhost:5000",
    "verbose": True,
    "max_iter": 10,
    "do_summary": True,
    "active_template": "default",
    "active_tools": ["DuckDuckGo", "Wikipedia"],
}

shared.templates = load_templates()
shared.active_template = params["active_template"]

shared.active_tools = set(params["active_tools"])


def create_agent():
    llm = OobaboogaLLM(params["api_endpoint"])
    return BaseAgent(
        "AutoLLaMa",
        shared.templates[shared.active_template],
        llm,
        [tool for tool in shared.tools if tool.name in shared.active_tools],
        verbose=params["verbose"],
    )


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
        template_tab()


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    if user_input[:3] == "/do":
        context_str = "Your reply should be based on this additional context:"

        objective = user_input.replace("/do", "").lstrip()
        res = create_agent().run(
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
