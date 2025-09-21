# from dotenv import load_dotenv
# from litellm import completion


# ## set ENV variables
# load_dotenv()


# # # res = supports_prompt_caching("gemini/gemini-2.0-flash")
# # # print(res)

# # # m n o pq r stu v wxyz

# # response = completion(
# #     model="gemini/gemini-2.5-flash",
# #     messages=[
# #         {
# #             "content": "Fill in the grap? m o r _ a, what is the value in this sequence _?",
# #             "role": "user",
# #         },
# #     ],
# #     reasoning_effort="low",
# #     thinking={"type": "enabled", "budget_tokens": 200},
# # )

# # # print(response)


# # # response = completion(
# # #     model="gemini/gemini-2.0-flash",
# # #     messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
# # #     thinking={"type": "enabled", "budget_tokens": 200},
# # # )
# # # print(response.id)
# # # print(response.usage)
# # # print(response.model_dump())


# # messages = [
# #     {"role": "user", "content": "List 5 popular cookie recipes."},
# #     {
# #         "role": "assistant",
# #         "content": "I can help you with that!, Can you share do you like tea?",
# #     },
# #     {"role": "user", "content": "I like it, share something"},
# # ]

# # response_schema = {
# #     "type": "array",
# #     "items": {
# #         "type": "object",
# #         "properties": {
# #             "recipe_name": {
# #                 "type": "string",
# #             },
# #         },
# #         "required": ["recipe_name"],
# #     },
# # }


# # response = completion(
# #     model="gemini/gemini-2.5-flash",
# #     messages=messages,
# #     response_format={
# #         "type": "json_object",
# #         "response_schema": response_schema,
# #     },  # 👈 KEY CHANGE
# #     safety_settings=[
# #         {
# #             "category": "HARM_CATEGORY_HARASSMENT",
# #             "threshold": "BLOCK_NONE",
# #         },
# #         {
# #             "category": "HARM_CATEGORY_HATE_SPEECH",
# #             "threshold": "BLOCK_NONE",
# #         },
# #         {
# #             "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
# #             "threshold": "BLOCK_NONE",
# #         },
# #         {
# #             "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
# #             "threshold": "BLOCK_NONE",
# #         },
# #     ],
# # )

# # print(response.model_dump())


# tools = [
#     {
#         "type": "function",
#         "function": {
#             "name": "get_current_weather",
#             "description": "Get the current weather in a given location",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "location": {
#                         "type": "string",
#                         "description": "The city and state, e.g. San Francisco, CA",
#                     },
#                     "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
#                 },
#                 "required": ["location"],
#             },
#         },
#     }
# ]
# messages = [
#     {"role": "user", "content": "What's the weather like in Boston today?"},
#     {
#         "role": "assistant",
#         "content": "",
#         "tool_calls": [
#             {
#                 "index": 0,
#                 "function": {
#                     "arguments": '{"location": "Boston, MA"}',
#                     "name": "get_current_weather",
#                 },
#                 "id": "call_de692ce5a11b49578c4c864e087a",
#                 "type": "function",
#             }
#         ],
#     },
#     {
#         "role": "tool",
#         "tool_call_id": "call_de692ce5a11b49578c4c864e087a",
#         "content": '{"temperature": "75°F", "condition": "Partly Cloudy", "humidity": "60%"}',
#     },
# ]

# messages = [
#     {"role": "system", "content": "You are a helpful assistant. For job"},
#     {
#         "role": "system",
#         "content": "Job ID: 12345, skills: [Python, Data Analysis] designation: Python developer",
#     },
#     {
#         "role": "system",
#         "content": (
#             "CV: Name shudipto, Skills: Python, Java, "
#             "Work Experience: 5 years in software development"
#         ),
#     },
#     {"role": "user", "content": "HI, I am ready"},
# ]

# response = completion(
#     model="gemini/gemini-2.5-flash",
#     messages=messages,
#     # tools=tools,
# )
# # Add any assertions, here to check response args
# # print(response.model_dump_json())  # Disabled for production
