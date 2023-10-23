import re, os
import pandas as pd

from enum import Enum
from requests import post

from extensions.auto_llama.llm import LLMInterface
from extensions.auto_llama.tool import (
    BaseTool,
    ActionStep,
    FinalStep,
    NoneTool,
)
from extensions.auto_llama.templates import (
    ToolChainTemplate,
    SummaryTemplate,
    ObjectiveTemplate,
    CodeTemplate,
)
import extensions.auto_llama.shared as shared


def is_active(agent: str):
    return agent in shared.active_agents


class AgentError(Exception):
    """Action chain failed"""

    pass


class AnswerType(Enum):
    """Different types of results from the AutoLLaMa Agent"""

    CONTEXT = "context"
    """ Text based result which should be added to the context """

    IMG = "img"
    """ Result Image which should be added to the response """

    CHAT = "chat"
    """ Text based result which should be added to the user input """

    PROMPT = "prompt"
    """ Text based result which replaces the current prompt in the chat"""

    RESPONSE = "response"
    """ Text based result which should be added to the response """


class SummaryAgent:
    """AutoLLaMa Agent which summarizes text"""


def __init__(
    self,
    name: str,
    prompt_template: SummaryTemplate,
    llm: LLMInterface,
    verbose: bool = False,
):
    self.name = name
    self.prompt_template = prompt_template
    self.llm = llm
    self.verbose = verbose


def run(self, objective: str, text: str) -> tuple[AnswerType, str]:
    print(f"> Running Agent: {self.name}")

    prompt = self.prompt_template.template.format(objective=objective, text=text)

    if self.verbose:
        print("Prompting LLM: ----------")
        print(prompt)

    summary = self.llm.completion(prompt, temperature=0.8, max_new_tokens=400)

    if self.verbose:
        print("Response: ----------")
        print(summary)

    return (AnswerType.RESPONSE, summary)


class CodeAgent:
    """Agent which is able to execute code"""

    DATA_PATH = os.path.join("code_exec", "files")
    allowed_filetypes = ["csv"]
    allowed_languages = ["python"]

    def __init__(
        self,
        name: str,
        prompt_template: CodeTemplate,
        llm: LLMInterface,
        pkg: list[str],
        executor_endpoint: str = "http://localhost:6000",
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.prompt_template = prompt_template
        self.llm = llm
        self.pkg = pkg
        self.data: dict[str, str] = {}
        self.executor_endpoint = executor_endpoint
        self.verbose = verbose

    def add_data(self, path: str):
        """Add data (.csv or similar) to the code executor"""

        basename = os.path.basename(path)
        file_type = basename.split(".")[-1].lower()

        if file_type not in self.allowed_filetypes:
            raise ValueError(f"Unsupported file type {file_type}")

        self.data[basename] = file_type

        # TODO: Move file into data folder of the container

    def add_pkg(self, *packages: str):
        """Extend list of usable python packages"""

        self.pkg.extend(packages)

        # TODO: Add to requirements.txt and install in container

    def _generate_file_prompt(self, file: tuple[str, str]):
        """Generate prompt for file with example"""

        prompt = f"{file[0]}:"

        if file[1] == "csv":
            df = pd.read_csv(file[0])

            # Load header and data types of each column in the csv
            cols = [f"{col}: {df[col].dtype}" for col in df.columns]

            prompt += " | ".join(cols)
            return prompt

    def _extract_code(self, text: str):
        """Extract code from llm response"""

        pattern = r"```(?P<language>.*)\n(?P<code>[^`]*)\n```"

        match = re.search(pattern, text)

        if match is None:
            raise ValueError("No code found in response")

        language = match.group("language")
        code = match.group("code").strip()

        return (language, code)

    def _execute_code(self, code: str):
        """Execute code in sandboxed environment and return output"""

        res = post(self.executor_endpoint, json={"code": code})

        if res.status_code != 200:
            raise AgentError("Failed to execute code")

        res_dict = res.json()

        return (res_dict["response"], res_dict["images"])

    def run(self, objective: str) -> list[tuple[AnswerType, str]]:
        print(f"> Running Agent: {self.name}")

        prompt = self.prompt_template.template.format(
            objective=objective,
            files="\n".join(
                [self._generate_file_prompt(file) for file in self.data.items()]
            ),
            packages=", ".join(self.pkg),
        )

        if self.verbose:
            print("Prompting LLM: ----------")
            print(prompt)

        result = self.llm.completion(prompt, max_new_tokens=800)

        if self.verbose:
            print("Response: ----------")
            print(result)

        try:
            lang, code = self._extract_code(prompt + result)
        except ValueError:
            return [(AnswerType.CHAT, "No valid code found in response")]

        if lang not in self.allowed_languages:
            return [(AnswerType.CHAT, code), (AnswerType.CHAT, f"Unsupported language {lang}")]

        try:
            output, images = self._execute_code(code)
        except AgentError:
            return [(AnswerType.CHAT, code), (AnswerType.CHAT, "Failed to execute code")]

        return [
            (AnswerType.CHAT, code),
            (AnswerType.CHAT, output),
            *[
                (AnswerType.IMG, f"{self.executor_endpoint}/image/{img}")
                for img in images
            ],
        ]


class ObjectiveAgent:
    """Agent which generates an simple objective from complex prompt"""

    def __init__(
        self,
        name: str,
        prompt_template: ObjectiveTemplate,
        llm: LLMInterface,
        tools: list[BaseTool],
        verbose: bool = False,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.llm = llm
        self.tools = tools
        self.verbose = verbose

    def run(self, text: str) -> tuple[AnswerType, str]:
        print(f"> Running Agent: {self.name}")

        prompt = self.prompt_template.template.format(
            text=text,
            tools="\n".join(
                [f"{tool.keywords[0]}: {tool.description}" for tool in self.tools]
            ),
        )

        if self.verbose:
            print("Prompting LLM: ----------")
            print(prompt)

        objective = self.llm.completion(prompt, max_new_tokens=100)

        if self.verbose:
            print("Response: ----------")
            print(objective)

        return (AnswerType.CHAT, objective)


class ToolChainAgent:
    """AutoLLaMa Agent which controls the Action chain"""

    def __init__(
        self,
        name: str,
        prompt_template: ToolChainTemplate,
        llm: LLMInterface,
        summary_agent: SummaryAgent,
        tools: list[BaseTool],
        verbose: bool = False,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.verbose = verbose
        self.llm = llm
        self.summary_agent = summary_agent
        self.tools = tools

    def run(
        self, objective: str, max_iter: int = 10, do_summary: bool = True
    ) -> tuple[AnswerType, str]:
        """Execute the action chain

        ARGUMENTS
            objective (str): Task/Question/Problem which should be solved by the Agent
            max_iter (int): Maximum iterations after which the chain exits automatically (Default: 10)
            do_summary (int): Whether the observations of A tool should be summarized. Reduces Absolute number of tokens in the prompt but increases Runtime (Default: True)

        RETURNS
            answer_type (AnswerType): Type of answer
            answer (str): The result of the action chain
        """

        print(f"> Running Agent: {self.name}")

        steps: list[ActionStep] = []

        for i in range(max_iter):
            if self.verbose:
                print(f"################# AutoLLaMa Step {i} #################")

            # Generate Prompt
            prompt = self._generate_prompt(objective, steps)

            if self.verbose:
                print("Prompting LLM: ----------")
                print(prompt)

            # Prompt LLM
            res = self.llm.completion(
                prompt,
                stopping_strings=[f"\n{self.prompt_template.observation_keyword}"],
            )

            if self.verbose:
                print("Response: ----------")
                print(res)

            # Parse response
            step = self._parse_output(res)

            # Action
            if step.is_final:
                print(f"> Final Answer found")

                if self.verbose:
                    print(step.observation)

                return (AnswerType.CONTEXT, step.observation)

            print(f">> Running Tool: {step.tool.name}")

            observation = step.tool.run(step.action_query, objective)

            if do_summary:
                print(f">>> Summarizing Results")
                _, observation = self.summary_agent.run(step.action_query, observation)

            step.set_observation(observation)

            steps.append(step)

        print("> Maximum Iterations Reached - Generating Final Answer")

        return (
            AnswerType.CONTEXT,
            self.summary_agent.run(
                objective, "\n\n".join((step.observation for step in steps))
            )[1],
        )

    def _generate_prompt(self, objective: str, steps: list[ActionStep]) -> str:
        tools_keywords = ", ".join([tool.keywords[0] for tool in self.tools])
        tools = "\n".join(
            [f"{tool.keywords[0]}: {tool.description}" for tool in self.tools]
        )

        agent_scratchpad = ""
        for thought, action, action_query, observation in (
            step.format() for step in steps
        ):
            agent_scratchpad += f"\n{self.prompt_template.thought_keyword}: {thought}"
            agent_scratchpad += f"\n{self.prompt_template.tool_keyword}: {action}"
            agent_scratchpad += (
                f"\n{self.prompt_template.tool_query_keyword}: {action_query}"
            )
            agent_scratchpad += (
                f"\n{self.prompt_template.observation_keyword}: {observation}"
            )

        agent_scratchpad += f"\n{self.prompt_template.thought_keyword}"

        return self.prompt_template.template.format(
            objective=objective,
            tools_keywords=tools_keywords,
            tools=tools,
            agent_scratchpad=agent_scratchpad,
        )

    def _parse_output(self, output: str) -> ActionStep:
        """Parse LLM output to ActionStep"""

        trunc_output = output.split(self.prompt_template.observation_keyword.lower())[
            0
        ].strip()

        if self.prompt_template.final_keyword in trunc_output:
            return FinalStep(trunc_output.split(self.prompt_template.final_keyword)[-1])

        thought_keyword = self.prompt_template.thought_keyword
        tool_keyword = self.prompt_template.tool_keyword
        tool_query_keyword = self.prompt_template.tool_query_keyword

        regex = rf"\s*\d*\s*:(.*?)\n{tool_keyword}\s*\d*\s*:(.*?)\n{tool_query_keyword}\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, trunc_output, re.DOTALL)

        if not match:
            return FinalStep(trunc_output)

        thought = match.group(1).strip()
        action = match.group(2).strip()
        action_query = match.group(3)

        for tool in self.tools:
            if tool.is_tool(action):
                return ActionStep(thought, tool, action_query)

        return ActionStep(thought, NoneTool(), action_query)
