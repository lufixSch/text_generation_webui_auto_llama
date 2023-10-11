import gradio as gr

import extensions.auto_llama.shared as shared
from extensions.auto_llama.agent import (
    ToolChainAgent,
    SummaryAgent,
    is_active as agent_is_active,
)
from extensions.auto_llama.templates import (
    ToolChainTemplate,
    SummaryTemplate,
    ObjectiveTemplate,
)
from extensions.auto_llama.config import (
    load_templates,
    save_templates,
    update_template,
    create_template,
    get_active_template,
)


def activate_template(name: str, agent: str, keys: list[str]):
    """Activate new template"""

    shared.active_templates[agent] = name

    if len(keys) > 1:
        return [
            gr.update(value=getattr(get_active_template(agent), key)) for key in keys
        ]

    return gr.update(value=getattr(get_active_template(agent), keys[0]))


def tool_chain_agent_tab():
    """Tab for updating/selecting prompt templates"""

    AGENT_NAME = "ToolChainAgent"

    with gr.Tab("Tool Chain Agent"):
        template = get_active_template(AGENT_NAME)

        template_textboxes: dict[str, gr.Textbox] = {}

        agent_active_checkbox = gr.Checkbox(
            value=agent_is_active(AGENT_NAME), label="Enable Agent"
        )

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates[AGENT_NAME].keys()],
            value=shared.active_templates[AGENT_NAME],
            label="Active Template",
            interactive=True,
        )

        with gr.Row():
            template_textboxes["tool_keyword"] = gr.Textbox(
                value=template.tool_keyword, label="Tool Keyword"
            )
            template_textboxes["tool_query_keyword"] = gr.Textbox(
                value=template.tool_query_keyword, label="Tool Input Keyword"
            )
            template_textboxes["observation_keyword"] = gr.Textbox(
                value=template.observation_keyword, label="Observation Keyword"
            )
            template_textboxes["thought_keyword"] = gr.Textbox(
                value=template.thought_keyword, label="Thought Keyword"
            )
            template_textboxes["final_keyword"] = gr.Textbox(
                value=template.final_keyword, label="Final Keyword"
            )

        template_textboxes["template"] = gr.TextArea(
            value=template.template, label="Template"
        )

        save_btn = gr.Button(value="Save Template")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New Template", interactive=False)

    agent_active_checkbox.change(
        lambda enable: shared.active_agents.add(AGENT_NAME)
        if enable
        else shared.active_agents.remove(AGENT_NAME),
        agent_active_checkbox,
        None,
    )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(
        lambda tool_keyword, tool_query_keyword, observation_keyword, thought_keyword, final_keyword, template: create_template(
            shared.active_templates[AGENT_NAME],
            AGENT_NAME,
            ToolChainTemplate(
                tool_keyword,
                tool_query_keyword,
                observation_keyword,
                thought_keyword,
                final_keyword,
                template,
            ),
        ),
        [*template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None)
    create_btn.click(
        lambda name, tool_keyword, tool_query_keyword, observation_keyword, thought_keyword, final_keyword, template: create_template(
            name,
            AGENT_NAME,
            ToolChainTemplate(
                tool_keyword,
                tool_query_keyword,
                observation_keyword,
                thought_keyword,
                final_keyword,
                template,
            ),
        ),
        [template_name_txt, *template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    ).then(
        lambda: gr.update(
            choices=[name for name in shared.templates[AGENT_NAME].keys()]
        ),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(name, AGENT_NAME, list(template_textboxes.keys())),
        template_choice,
        [*template_textboxes.values()],
    )


def summary_agent_tab():
    """Tab for personalizing settings for the summary Agent"""

    AGENT_NAME = "SummaryAgent"

    with gr.Tab("Summary Agent"):
        template = get_active_template(AGENT_NAME)

        template_textboxes: dict[str, gr.Textbox] = {}

        agent_active_checkbox = gr.Checkbox(
            value=agent_is_active(AGENT_NAME), label="Enable Agent"
        )

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates[AGENT_NAME].keys()],
            value=shared.active_templates[AGENT_NAME],
            label="Active Template",
            interactive=True,
        )

        template_textboxes["prefix"] = gr.Textbox(
            value=template.prefix, label="Summary Prefix"
        )

        template_textboxes["template"] = gr.TextArea(
            value=template.template, label="Template"
        )

        save_btn = gr.Button(value="Save Template")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New Template", interactive=False)

    agent_active_checkbox.change(
        lambda enable: shared.active_agents.add(AGENT_NAME)
        if enable
        else shared.active_agents.remove(AGENT_NAME),
        agent_active_checkbox,
        None,
    )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(
        lambda prefix, template: create_template(
            shared.active_templates[AGENT_NAME],
            AGENT_NAME,
            SummaryTemplate(prefix, template),
        ),
        [*template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None)
    create_btn.click(
        lambda name, prefix, template: create_template(
            name,
            AGENT_NAME,
            SummaryTemplate(prefix, template),
        ),
        [template_name_txt, *template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    ).then(
        lambda: gr.update(
            choices=[name for name in shared.templates[AGENT_NAME].keys()]
        ),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(name, AGENT_NAME, list(template_textboxes.keys())),
        template_choice,
        [*template_textboxes.values()],
    )


def objective_agent_tab():
    """Tab for personalizing settings for the ObjectiveAgent"""

    AGENT_NAME = "ObjectiveAgent"

    with gr.Tab("Objective Agent"):
        template = get_active_template(AGENT_NAME)

        template_textboxes: dict[str, gr.Textbox] = {}

        agent_active_checkbox = gr.Checkbox(
            value=agent_is_active(AGENT_NAME), label="Enable Agent"
        )

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates[AGENT_NAME].keys()],
            value=shared.active_templates[AGENT_NAME],
            label="Active Template",
            interactive=True,
        )

        template_textboxes["template"] = gr.TextArea(
            value=template.template, label="Template"
        )

        save_btn = gr.Button(value="Save Template")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New Template", interactive=False)

    agent_active_checkbox.change(
        lambda enable: shared.active_agents.add(AGENT_NAME)
        if enable
        else shared.active_agents.remove(AGENT_NAME),
        agent_active_checkbox,
        None,
    )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(
        lambda template: create_template(
            shared.active_templates[AGENT_NAME],
            AGENT_NAME,
            ObjectiveTemplate(template),
        ),
        [*template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None)
    create_btn.click(
        lambda name, template: create_template(
            name,
            AGENT_NAME,
            ObjectiveTemplate(template),
        ),
        [template_name_txt, *template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    ).then(
        lambda: gr.update(
            choices=[name for name in shared.templates[AGENT_NAME].keys()]
        ),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(name, AGENT_NAME, list(template_textboxes.keys())),
        template_choice,
        [*template_textboxes.values()],
    )


def tool_tab():
    """Tab for disabling/enabling tools"""

    tool_choice: list[gr.Checkbox] = []

    with gr.Tab("Tools"):
        tool = shared.tools[0]
        tool_choice.append(
            gr.Checkbox(
                value=tool.name in shared.active_tools,
                label=tool.name,
                interactive=True
            )
        )

        tool = shared.tools[1]
        tool_choice.append(
            gr.Checkbox(
                value=tool.name in shared.active_tools,
                label=tool.name,
                interactive=True
            )
        )


    tool = shared.tools[0]
    tool_choice[0].change(
        lambda active: shared.active_tools.add(shared.tools[0].name)
        if active
        else shared.active_tools.remove(shared.tools[0].name),
        tool_choice[0],
        None,
    )

    tool = shared.tools[1]
    tool_choice[1].change(
        lambda active: shared.active_tools.add(shared.tools[1].name)
        if active
        else shared.active_tools.remove(shared.tools[1].name),
        tool_choice[1],
        None,
    )
