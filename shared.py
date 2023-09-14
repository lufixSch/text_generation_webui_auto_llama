from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool, BaseTool
from extensions.auto_llama.agent import BaseAgent, PromptTemplate
from extensions.auto_llama.llm import LLMInterface

templates: dict[str, PromptTemplate] = {}
active_template: str = "default"

tools: list[BaseTool] = [WikipediaTool(max_articles=2), DuckDuckGoSearchTool(max_results=10)]
active_tools: set[str] = []

llm: LLMInterface = None
agent: BaseAgent = None
