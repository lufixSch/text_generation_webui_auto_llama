import gradio as gr

import extensions.auto_llama.shared as shared
from extensions.auto_llama.config import (
    load_templates,
    save_templates,
    update_template,
    create_template,
)


def activate_template(name: str, keys: list[str]):
    """Activate new template"""

    shared.active_template = name

    return [gr.update(value=getattr(templates[active_template], key)) for key in keys]


def template_tab():
    """Tab for updating/selecting prompt templates"""

    with gr.Tab("Prompt"):
        template = shared.templates[shared.active_template]
        template_textboxes: dict[str, gr.Textbox] = {}

        template_choice = gr.Dropdown(
            choices=[name for name in shared.templates.keys()],
            value=shared.active_template,
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

    template_textboxes["tool_keyword"].change(
        lambda txt: update_template(active_template, "tool_keyword", txt),
        template_textboxes["tool_keyword"],
        None,
    )
    template_textboxes["tool_query_keyword"].change(
        lambda txt: update_template(active_template, "tool_query_keyword", txt),
        template_textboxes["tool_query_keyword"],
        None,
    )
    template_textboxes["observation_keyword"].change(
        lambda txt: update_template(active_template, "observation_keyword", txt),
        template_textboxes["observation_keyword"],
        None,
    )
    template_textboxes["thought_keyword"].change(
        lambda txt: update_template(active_template, "thought_keyword", txt),
        template_textboxes["thought_keyword"],
        None,
    )
    template_textboxes["final_keyword"].change(
        lambda txt: update_template(active_template, "final_keyword", txt),
        template_textboxes["final_keyword"],
        None,
    )
    template_textboxes["template"].change(
        lambda txt: update_template(active_template, "template", txt),
        template_textboxes["template"],
        None,
    )

    template_name_txt.change(
        lambda txt: gr.update(interactive=True)
        if txt != ""
        else gr.update(interactive=False),
        template_name_txt,
        create_btn,
    )

    save_btn.click(lambda: save_templates(shared.templates), None, None)
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
        [template_name_txt, *template_textboxes.values()],
        None,
    ).then(lambda: save_templates(shared.templates), None, None).then(
        lambda: gr.update(value=""), None, template_name_txt
    ).then(
        lambda: gr.update(choices=[name for name in shared.templates.keys()]),
        None,
        template_choice,
    )

    template_choice.select(
        lambda name: activate_template(name, template_textboxes.keys()),
        template_choice,
        [*template_textboxes.values()],
    )


def tool_tab():
    """Tab for disabling/enabling tools"""

    tool_choice: list[gr.Checkbox] = []

    with gr.Tab("Tools"):
        for tool in shared.tools:
            checkbox = gr.Checkbox(
                value=tool.name in shared.active_tools,
                label=tool.name,
                interactive=False,
                elem_id=tool.name
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