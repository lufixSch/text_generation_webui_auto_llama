class ToolChainTemplate:
    """Prompt template information for the ToolChainAgent"""

    def __init__(
        self,
        tool_keyword: str,
        tool_query_keyword: str,
        observation_keyword: str,
        thought_keyword: str,
        final_keyword: str,
        template: str,
    ):
        self.tool_keyword = tool_keyword
        self.tool_query_keyword = tool_query_keyword
        self.observation_keyword = observation_keyword
        self.thought_keyword = thought_keyword
        self.final_keyword = final_keyword
        self.template = template


class SummaryTemplate:
    """Prompt template information for the SummaryAgent"""

    def __init__(
        self,
        prefix: str,
        template: str,
    ):
        self.prefix = prefix
        self.template = template


class ObjectiveTemplate:
    """Prompt template information for the ObjectiveAgent"""

    def __init__(self, template: str):
        self.template = template


class CodeTemplate:
    """Prompt template information for the CodeAgent"""

    def __init__(self, template: str) -> None:
        self.template = template
