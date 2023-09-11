from abc import ABC, abstractmethod

from langchain.utilities import WikipediaAPIWrapper


class ActionStep:
    """One step in the action chain"""

    def __init__(
        self, thought: str, tool: "BaseTool", action_query: str, is_final: bool = False
    ):
        self.thought = thought
        self.tool = tool
        self.action_query = action_query
        self.observation = ""
        self.is_final = is_final

    def format(self) -> tuple[str, str, str, str]:
        return (
            self.thought,
            self.tool.keywords[0],
            self.action_query,
            self.observation,
        )

    def set_observation(self, observation: str):
        self.observation = observation


class FinalStep(ActionStep):
    """Final step in chain"""

    def __init__(self, observation: str):
        super().__init__(None, None, None, is_final=True)
        self.set_observation(observation)


class BaseTool(ABC):
    """Tool which the LLaMa Agent can use"""

    name: str
    func: callable
    description: str
    keywords: list[str]

    def __init__(self, name: str, description: str, keywords: list[str] | str = None):
        if not keywords:
            keywords = name

        if isinstance(keywords, str):
            keywords = [keywords]

        self.name = name
        self.description = description
        self.keywords = keywords

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute Tool"""

        raise NotImplementedError("Every tool needs to implement the `run` method")

    def is_tool(self, action_query: str) -> bool:
        """Check if this tool is meant by the action query"""

        for keyword in self.keywords:
            if keyword in action_query:
                return True

        return False


class NoneTool(BaseTool):
    """Fallback tool, when no matching tool is found"""

    def __init__(self):
        super().__init__("None", "No matching tool was found")

    def run(self, query: str) -> str:
        return "No tool was found to perform this Action"


class WikipediaTool(BaseTool):
    """Search wikipedia"""

    wiki_api = WikipediaAPIWrapper()

    def __init__(self):
        super().__init__(
            "Wikipedia",
            description="Wikipedia serves as a versatile tool, offering uses such as gathering background information, exploring unfamiliar topics, finding reliable sources, understanding current events, discovering new interests, and obtaining a comprehensive overview on diverse subjects like historical events, scientific concepts, biographies of notable individuals, geographical details, cultural phenomena, artistic works, technological advancements, social issues, academic subjects, making it a valuable resource for learning and knowledge acquisition.",
            keywords=["search", "Search", "Wikipedia", "wikipedia"],
        )

    def run(self, query: str) -> str:
        return self.wiki_api.run(query)
