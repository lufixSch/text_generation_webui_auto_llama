from abc import ABC, abstractmethod
from itertools import islice

import wikipedia
import wolframalpha
from duckduckgo_search import DDGS

from extensions.auto_llama.llm import LLMInterface

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
    def run(self, query: str, objective: str) -> str:
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

    def run(self, query: str, _:str) -> str:
        return "No tool was found to perform this Action"


class WikipediaTool(BaseTool):
    """Search wikipedia"""

    def __init__(self, max_articles: int=1):
        self.max_articles = max_articles
        
        super().__init__(
            "Wikipedia",
            description="Wikipedia serves as a versatile tool, offering uses such as gathering background information, exploring unfamiliar topics, finding reliable sources, understanding current events, discovering new interests, and obtaining a comprehensive overview on diverse subjects like historical events, scientific concepts, biographies of notable individuals, geographical details, cultural phenomena, artistic works, technological advancements, social issues, academic subjects, making it a valuable resource for learning and knowledge acquisition.",
            keywords=["learn", "Learn", "discover", "Discover", "Wikipedia", "wikipedia"],
        )

    def run(self, query: str, _:str) -> str:
        articles = wikipedia.search(query)[:self.max_articles]
        
        summaries = []
        for article in articles:
            try:
                summary = wikipedia.summary(article, auto_suggest=False)
            except wikipedia.PageError as err:
                continue
            
            summaries.append(f"{article}\n{summary}")
            
        if not summaries:
            return "No good Wikipedia Search Result was found"
            
        return "\n\n".join(summaries)


class DuckDuckGoSearchTool(BaseTool):
    """ Search DuckDuckGo """
    
    def __init__(self, max_results: int=3):
        self.max_results = max_results
        
        super().__init__(
            "DuckDuckGo",
            description="The DuckDuckGo search engine is a tool designed to find information on the Internet by searching and retrieving web pages that contain the desired information. It function as an index to the billions of web pages available on the internet and allows you to find information about specific topics or recent events, news, documentation of programs, user generated content like forums or blogs. Inputs are keywords or phrases related to their topic of interest, and the search engine will display results based on relevancy",
            keywords=["search", "Search", "find", "Find", "duckduckgo", "DuckDuckGo"]
        )
        
    def run(self, query: str, _: str) -> str:
        with DDGS() as ddgs:
            results = ""
            
            for t in islice(ddgs.text(query), self.max_results):
                results += "\n\n" + t['title'] + "\n" + t['body'] + "\nSource: " + t['href']

            if results == "":
                return "No good DuckDuckGo Search Result was found"

            return results
        