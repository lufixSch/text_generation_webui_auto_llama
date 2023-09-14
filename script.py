import gradio as gr

from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool
from extensions.auto_llama.agent import BaseAgent, PromptTemplate
from extensions.auto_llama.llm import OobaboogaLLM
from extensions.auto_llama.config import load_templates, save_templates

from modules import chat, extensions

extension_name = "auto_llama"

params = {
    "display_name": "AutoLLaMa",
    "api_endpoint": "http://localhost:5000",
    "verbose": True,
    "max_iter": 10,
    "do_summary": True,
    "active_template": "default",
}

templates: dict[str, PromptTemplate] = load_templates()
active_template = params["active_template"]

tools = [WikipediaTool(), DuckDuckGoSearchTool()]

llm = None
agent = None


def update_template(name: str, key: str, value: str):
    """Update shared templates"""

    setattr(templates[name], key, value)


def create_template(name: str, template: PromptTemplate):
    """Create new template"""

    templates[name] = template


def activate_template(name: str, keys: list[str]):
    """Activate new template"""

    global active_template
    active_template = name
    
    return [gr.update(value=getattr(templates[active_template], key)) for key in keys]


def setup():
    """Gets executed when the extension is first loaded"""

    llm = OobaboogaLLM(params["api_endpoint"])
    agent = BaseAgent(
        "AutoLLaMa", templates[active_template], llm, tools, params["verbose"]
    )


def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.
    """

    template = templates[active_template]
    template_textboxes: dict[str, gr.Textbox] = {}

    # with gr.Box(visible=False, elem_classes="file-saver") as file_saver:
    #    with gr.Row():
    #        template_name_txt = gr.Textbox(value=active_template, label="Template Name")
    #        save_new_btn = gr.Button(value="Save")

    with gr.Accordion("AutoLLaMa", open=False):
        # template_select = gr.Dropdown(choices=[name for name in params['templates'].keys()])

        template_choice = gr.Dropdown(
            choices=[name for name in templates.keys()],
            value=params["active_template"],
            label="Active Template",
            interactive=True
        )

        with gr.Row():
            template_textboxes['tool_keyword'] = gr.Textbox(
                value=template.tool_keyword, label="Tool Keyword"
            )
            template_textboxes['tool_query_keyword'] = gr.Textbox(
                value=template.tool_query_keyword, label="Tool Input Keyword"
            )
            template_textboxes['observation_keyword'] = gr.Textbox(
                value=template.observation_keyword, label="Observation Keyword"
            )
            template_textboxes['thought_keyword'] = gr.Textbox(
                value=template.thought_keyword, label="Thought Keyword"
            )
            template_textboxes['final_keyword'] = gr.Textbox(
                value=template.final_keyword, label="Final Keyword"
            )

        template_textboxes['template'] = gr.TextArea(value=template.template, label="Template")

        save_btn = gr.Button(value="Save")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New", interactive=False)

    template_textboxes['tool_keyword'].change(
        lambda txt: update_template(active_template, "tool_keyword", txt),
        template_textboxes['tool_keyword'],
        None,
    )
    template_textboxes['tool_query_keyword'].change(
        lambda txt: update_template(active_template, "tool_query_keyword", txt),
        template_textboxes['tool_query_keyword'],
        None,
    )
    template_textboxes['observation_keyword'].change(
        lambda txt: update_template(active_template, "observation_keyword", txt),
        template_textboxes['observation_keyword'],
        None,
    )
    template_textboxes['thought_keyword'].change(
        lambda txt: update_template(active_template, "thought_keyword", txt),
        template_textboxes['thought_keyword'],
        None,
    )
    template_textboxes['final_keyword'].change(
        lambda txt: update_template(active_template, "final_keyword", txt),
        template_textboxes['final_keyword'],
        None,
    )
    template_textboxes['template'].change(
        lambda txt: update_template(active_template, "template", txt),
        template_textboxes['template'],
        None,
    )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(lambda: save_templates(templates), None, None)
    # create_btn.click(lambda: gr.update(visible=True), None, file_saver)
    create_btn.click(
        lambda name, tool_keyword, tool_query_keyword, observation_keyword, thought_keyword, final_keyword, template: create_template(
            name,
            PromptTemplate(
                tool_keyword,
                tool_query_keyword,
                observation_keyword,
                thought_keyword,
                final_keyword,
                template,
            ),
        ),
        [
            template_name_txt,
            *template_textboxes.values()
        ],
        None,
    ).then(lambda: save_templates(templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    )
    
    template_choice.select(lambda name: activate_template(name, template_textboxes.keys()), template_choice, [*template_textboxes.values()])
    


def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """

    if user_input[:3] == "/do":
        context_str = "Your reply should be based on this additional context:"

        objective = user_input.replace("/do", "").lstrip()
        res = BaseAgent(
            "AutoLLaMa",
            get_template(params["active_template"]),
            llm,
            tools,
            verbose=params["verbose"],
        ).run(objective, max_iter=params["max_iter"], do_summary=params["do_summary"])

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
