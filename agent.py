import re

from enum import Enum

from extensions.auto_llama.llm import LLMInterface
from extensions.auto_llama.tool import (
    BaseTool,
    ActionStep,
    FinalStep,
    NoneTool,
    SummarizeTool,
)


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


class PromptTemplate:
    """Prompt template information"""

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


class BaseAgent:
    """AutoLLaMa Agent which controls the Action chain"""

    summary_prefix = (
        "When answering the question consider the following additional information:"
    )

    def __init__(
        self,
        name: str,
        prompt_template: PromptTemplate,
        llm: LLMInterface,
        tools: list[BaseTool],
        verbose: bool = False,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.verbose = verbose
        self.llm = llm
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
        summarize_tool: SummarizeTool = SummarizeTool(self.llm)

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
                stopping_strings=[f"\n{self.prompt_template.observation_keyword}"]
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
                observation = summarize_tool.run(observation, step.action_query)

            step.set_observation(observation)

            steps.append(step)

        print("> Maximum Iterations Reached - Generating Final Answer")

        return (
            AnswerType.CONTEXT,
            summarize_tool.run("\n\n".join((step.observation for step in steps)), objective),
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
