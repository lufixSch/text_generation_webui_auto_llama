# AutoLLaMa (An Oobabooga text-generation-webui Extension)

Run an Agent in the background which uses different tools to solve a given problem/question.

The Agent is triggered in the Chat mode by starting a message with `/do`. The Agent will then use the provided tools to solve the given problem and will add the results to the chat context.

## Tools
### Existing Tools
- Wikipedia

### Adding Tools

New tools can be added by creating a new class which extends the `LLMInterface` class in the `llm` module. An instance of the new class must be added to the list of `tools` in `script.py`

## Installation

```bash
cd extensions
clone https://github.com/lufixSch/text_generation_webui_auto_llama_.git auto_llama
python server.py --extensions auto_llama api
```

> AutoLLaMa requires the API to be active. Make sure the API extension is enabled