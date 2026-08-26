from openai import OpenAI

client = OpenAI(
    base_url="https://modal-labs--ep-glm-5-2-fp8-server.us-west.modal.direct/v1",
    api_key="unused",
)

completion = client.chat.completions.create(
    model="zai-org/GLM-5.2-FP8",
    messages=[
        {
            "role": "system",
            "content": "You are a concise technical assistant.",
        },
        {
            "role": "user",
            "content": "What is the weather in Paris?",
        },
    ],
    temperature=0.3,
    max_tokens=2048,
    top_p=0.9,
    stream=False,
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
    ],
    tool_choice="auto",
    extra_body={"reasoning_effort": "none"},
)
print(completion)
print(completion.choices[0].message.content)