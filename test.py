from dotenv import load_dotenv
from litellm import completion
import os

from litellm.utils import supports_prompt_caching

## set ENV variables
load_dotenv()


res = supports_prompt_caching("gemini/gemini-2.0-flash")
print(res)

# m n o pq r stu v wxyz

# response = completion(
#     model="gemini/gemini-2.5-flash",
#     messages=[
#         {
#             "content": "Fill in the grap? m o r _ a, what is the value in this sequence _?",
#             "role": "user",
#         },
#     ],
#     reasoning_effort="low",
#     thinking={"type": "enabled", "budget_tokens": 200},
# )

# print(response)


tools = [{"googleSearch": {}}]  # 👈 ADD GOOGLE SEARCH

response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
    tools=tools,
)
print(response)
