import docker

from extensions.auto_llama.tool import WikipediaTool, DuckDuckGoSearchTool, BaseTool
from extensions.auto_llama.agent import ToolChainAgent, SummaryAgent, AnswerType, ObjectiveAgent, CodeAgent
from extensions.auto_llama.templates import ToolChainTemplate, SummaryTemplate, ObjectiveTemplate, CodeTemplate
from extensions.auto_llama.llm import LLMInterface

templates: dict[str, dict[str, ToolChainTemplate | SummaryTemplate | ObjectiveTemplate | CodeTemplate]] = {}
active_templates: dict[str, str] = {}

tools: list[BaseTool] = [WikipediaTool(max_articles=2), DuckDuckGoSearchTool(max_results=10)]
active_tools: set[str] = []
allowed_packages: set[str] = []

llm: LLMInterface = None
agents: dict[str, ToolChainAgent | SummaryAgent | ObjectiveAgent | CodeAgent] = None

active_agents: set[str] = []

response_modifier: list[tuple[AnswerType, any]] = []
""" Agent responses which should be added to the response. """

code_agent: CodeAgent = None

docker_client = docker.from_env()