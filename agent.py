import re

from enum import Enum

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
