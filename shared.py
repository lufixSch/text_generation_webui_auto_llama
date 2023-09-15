from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool, BaseTool
from extensions.auto_llama.agent import ToolChainAgent, SummaryAgent
from extensions.auto_llama.templates import ToolChainTemplate, SummaryTemplate
from extensions.auto_llama.llm import LLMInterface

templates: dict[str, dict[str, ToolChainTemplate | SummaryTemplate]] = {}
active_templates: dict[str, str] = {}

tools: list[BaseTool] = [WikipediaTool(max_articles=2), DuckDuckGoSearchTool(max_results=10)]
active_tools: set[str] = []

llm: LLMInterface = None
agents: dict[str, ToolChainAgent | SummaryAgent] = None
