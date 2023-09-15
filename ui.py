import gradio as gr

import extensions.auto_llama.shared as shared
from extensions.auto_llama.agent import ToolChainAgent, SummaryAgent
from extensions.auto_llama.templates import ToolChainTemplate, SummaryTemplate
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
        return [gr.update(value=getattr(get_active_template(agent), key)) for key in keys]
    
    return gr.update(value=getattr(get_active_template(agent), keys[0]))


def tool_chain_agent_tab():
    """Tab for updating/selecting prompt templates"""

    with gr.Tab("Tool Chain Agent"):
        template = get_active_template("ToolChainAgent")

        template_textboxes: dict[str, gr.Textbox] = {}

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates["ToolChainAgent"].keys()],
            value=shared.active_templates["ToolChainAgent"],
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

        save_btn = gr.Button(value="Save")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New", interactive=False)

    # template_textboxes["tool_keyword"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"],
    #         "ToolChainAgent",
    #         "tool_keyword",
    #         txt,
    #     ),
    #     template_textboxes["tool_keyword"],
    #     None,
    # )
    # template_textboxes["tool_query_keyword"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"],
    #         "ToolChainAgent",
    #         "tool_query_keyword",
    #         txt,
    #     ),
    #     template_textboxes["tool_query_keyword"],
    #     None,
    # )
    # template_textboxes["observation_keyword"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"],
    #         "ToolChainAgent",
    #         "observation_keyword",
    #         txt,
    #     ),
    #     template_textboxes["observation_keyword"],
    #     None,
    # )
    # template_textboxes["thought_keyword"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"],
    #         "ToolChainAgent",
    #         "thought_keyword",
    #         txt,
    #     ),
    #     template_textboxes["thought_keyword"],
    #     None,
    # )
    # template_textboxes["final_keyword"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"],
    #         "ToolChainAgent",
    #         "final_keyword",
    #         txt,
    #     ),
    #     template_textboxes["final_keyword"],
    #     None,
    # )
    # template_textboxes["template"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["ToolChainAgent"], "ToolChainAgent", "template", txt
    #     ),
    #     template_textboxes["template"],
    #     None,
    # )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(
        lambda tool_keyword, tool_query_keyword, observation_keyword, thought_keyword, final_keyword, template: create_template(
            shared.active_templates["ToolChainAgent"],
            "ToolChainAgent",
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
            "ToolChainAgent",
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
            choices=[name for name in shared.templates["ToolChainAgent"].keys()]
        ),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(
            name, "ToolChainAgent", template_textboxes.keys()
        ),
        template_choice,
        [*template_textboxes.values()],
    )


def summary_agent_tab():
    """Tab for personalizing settings for the summary Agent"""

    with gr.Tab("Summary Agent"):
        template = get_active_template("SummaryAgent")

        template_textboxes: dict[str, gr.Textbox] = {}

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates["SummaryAgent"].keys()],
            value=shared.active_templates["SummaryAgent"],
            label="Active Template",
            interactive=True,
        )

        template_textboxes["prefix"] = gr.Textbox(
            value=template.prefix, label="Summary Prefix"
        )

        template_textboxes["template"] = gr.TextArea(
            value=template.template, label="Template"
        )

        save_btn = gr.Button(value="Save")
        with gr.Row():
            template_name_txt = gr.Textbox(
                placeholder="Name of the new template", show_label=False, max_lines=1
            )
            create_btn = gr.Button(value="Create New", interactive=False)

    # template_textboxes["prefix"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["SummaryAgent"], "SummaryAgent", "prefix", txt
    #     ),
    #     template_textboxes["prefix"],
    #     None
    # )
    # template_textboxes["template"].change(
    #     lambda txt: update_template(
    #         shared.active_templates["SummaryAgent"], "SummaryAgent", "template", txt
    #     ),
    #     template_textboxes["template"],
    #     None,
    # )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(
        lambda prefix, template: create_template(
            shared.active_templates["SummaryAgent"],
            "SummaryAgent",
            SummaryTemplate(prefix, template),
        ),
        [*template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None)
    create_btn.click(
        lambda name, prefix, template: create_template(
            name,
            "SummaryAgent",
            SummaryTemplate(prefix, template),
        ),
        [template_name_txt, *template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    ).then(
        lambda: gr.update(
            choices=[name for name in shared.templates["SummaryAgent"].keys()]
        ),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(name, "SummaryAgent", template_textboxes.keys()),
        template_choice,
        [*template_textboxes.values()],
    )


def objective_agent_tab():
    """Tab for personalizing settings for the summary Agent"""
    

def tool_tab():
    """Tab for disabling/enabling tools"""

    tool_choice: list[gr.Checkbox] = []

    with gr.Tab("Tools"):
        for tool in shared.tools:
            checkbox = gr.Checkbox(
                value=tool.name in shared.active_tools,
                label=tool.name,
                interactive=False,
                elem_id=tool.name,
            )
            tool_choice.append(checkbox)

    for i, tool in enumerate(shared.tools):
        tool_choice[i].change(
            lambda active: shared.active_tools.add(tool.name)
            if active
            else shared.active_tools.remove(tool.name),
            checkbox,
            None,
        ).then(lambda: print(shared.active_tools)).then(lambda: print(tool.name))
