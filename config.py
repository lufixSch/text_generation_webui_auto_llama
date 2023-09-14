import os, json
from extensions.auto_llama.agent import PromptTemplate

_BASE_PATH = os.path.dirname(os.path.abspath(__file__))

def load_templates() -> dict[str, PromptTemplate]:
    """load templates"""
    
    with open(os.path.join(_BASE_PATH, 'templates.json')) as f:
        template_dict: dict[str, dict] = json.load(f)
        
    return {key: PromptTemplate(**vals) for key, vals in template_dict.items()}


def save_templates(templates: dict[str, PromptTemplate]):
    """Save templates to templates.json"""
    
    template_dict = {key: val.__dict__ for key, val in templates.items()}
    
    with open(os.path.join(_BASE_PATH, 'templates.json'), mode='w') as f:
        json.dump(template_dict, f)