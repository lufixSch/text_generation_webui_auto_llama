import os, json

import extensions.auto_llama.shared as shared
from extensions.auto_llama.templates import ToolChainTemplate, SummaryTemplate, ObjectiveTemplate, CodeTemplate

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def get_active_template(agent: str):
    """Return active template of a given agent"""

    return shared.templates[agent][shared.active_templates[agent]]


def update_template(name: str, agent: str, key: str, value: str):
    """Update shared templates"""

    setattr(shared.templates[agent][name], key, value)


def create_template(
    name: str, agent: str, template: ToolChainTemplate | SummaryTemplate | ObjectiveTemplate
):
    """Create new template"""

    shared.templates[agent][name] = template


def load_templates() -> dict[str, dict[ToolChainTemplate | SummaryTemplate | ObjectiveTemplate]]:
    """load templates"""

    with open(os.path.join(_BASE_PATH, "templates.json")) as f:
        template_dict: dict[str, dict] = json.load(f)

    return {
        "ToolChainAgent": {
            key: ToolChainTemplate(**vals)
            for key, vals in template_dict["ToolChainAgent"].items()
        },
        "SummaryAgent": {
            key: SummaryTemplate(**vals)
            for key, vals in template_dict["SummaryAgent"].items()
        },
        "ObjectiveAgent": {
            key: ObjectiveTemplate(**vals)
            for key, vals in template_dict["ObjectiveAgent"].items()
        },
        "CodeAgent": {
            key: CodeTemplate(**vals)
            for key, vals in template_dict['CodeAgent'].items()
        }
    }


def save_templates(
    templates: dict[str, dict[str, ToolChainTemplate | SummaryTemplate | ObjectiveTemplate]]
):
    """Save templates to templates.json"""

    template_dict = {
        key: {name: val.__dict__ for name, val in template.items()}
        for key, template in templates.items()
    }

    with open(os.path.join(_BASE_PATH, "templates.json"), mode="w") as f:
        json.dump(template_dict, f)
