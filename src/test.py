from dotenv import load_dotenv
load_dotenv()

from gen_ai_hub.proxy.native.openai import chat

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
    {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
    {"role": "user", "content": "Do other Azure Cognitive Services support this too?"},
]

kwargs = dict(model_name="gpt-4.1", messages=messages)
response = chat.completions.create(**kwargs)

# print(response)
print(response.choices[0].message.content)
