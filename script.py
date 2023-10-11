import gradio as gr


import extensions.auto_llama.shared as shared

from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool
from extensions.auto_llama.agent import (
    ToolChainAgent,
    SummaryAgent,
    ObjectiveAgent,
    AnswerType,
    is_active as agent_is_active,
)
from extensions.auto_llama.llm import OobaboogaLLM
from extensions.auto_llama.config import load_templates, get_active_template
from extensions.auto_llama.ui import (
    tool_chain_agent_tab,
    tool_tab,
    summary_agent_tab,
    objective_agent_tab,
)

from modules import chat, extensions

extension_name = "auto_llama"

params = {
    "display_name": "AutoLLaMa",
    "api_endpoint": "http://localhost:5000",
    "verbose": True,
    "max_iter": 10,
    "active_templates": {
        "ToolChainAgent": "default",
        "SummaryAgent": "default",
        "ObjectiveAgent": "default",
    },
    "active_tools": ["DuckDuckGo", "Wikipedia"],
    "active_agents": ["ToolChainAgent", "SummaryAgent", "ObjectiveAgent"],
}


def create_objective_agent():
    return ObjectiveAgent(
        "ObjectiveAgent",
        get_active_template("ObjectiveAgent"),
        shared.llm,
        [tool for tool in shared.tools if tool.name in shared.active_tools],
        verbose=params["verbose"],
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

    return create_objective_agent().run(chat_messages)


def setup():
    shared.templates = load_templates()

    shared.active_templates = params["active_templates"]
    shared.active_tools = set(params["active_tools"])
    shared.active_agents = set(params["active_agents"])

    shared.llm = OobaboogaLLM(params["api_endpoint"])


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.
    """

    with gr.Accordion("AutoLLaMa", open=False):
        tool_tab()
        tool_chain_agent_tab()
        summary_agent_tab()
        objective_agent_tab()


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    if user_input[:3] == "/do":
        context_str = "Your reply should be based on this additional context:"

        user_input = user_input.replace("/do", "").lstrip()

        if agent_is_active("ObjectiveAgent"):
            answer_type, objective = generate_objective(
                user_input, state["history"]["visible"]
            )
        else:
            objective = user_input

        if agent_is_active("ToolChainAgent"):
            answer_type, res = create_tool_chain_agent().run(
                objective,
                max_iter=params["max_iter"],
                do_summary=agent_is_active("SummaryAgent"),
            )
        else:
            res = objective

        if answer_type == AnswerType.CONTEXT:
            old_context = str(state["context"]).strip()
            context_already_included = context_str in old_context

            state["context"] = (
                old_context
                + (
                    "\n"
                    if context_already_included
                    else "\n\nYour reply should be based on this additional context:\n"
                )
                + res
                + "\n"
            )
        elif answer_type == AnswerType.CHAT:
            user_input = res
        elif answer_type == AnswerType.IMG:
            raise NotImplementedError("IMG AnswerType not implemented yet")
        elif answer_type == AnswerType.RESPONSE:
            raise NotImplementedError("RESPONSE AnswerType not implemented yet")
        else:
            raise ValueError(f"AnswerType {answer_type} not found")

    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result
