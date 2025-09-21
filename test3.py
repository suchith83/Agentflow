# from inspect import (
#     isasyncgen,
#     isasyncgenfunction,
#     isawaitable,
#     iscoroutinefunction,
#     isgenerator,
#     isgeneratorfunction,
# )

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
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "What is the weather like in Boston today?"},
# ]

# response = completion(
#     model="gemini/gemini-2.5-flash",
#     messages=messages,
#     tools=tools,
#     stream=True,
# )

# print(type(response))

# # for chunk in response:
# #     msg: ModelResponseStream = chunk  # type: ignore
# #     print(msg.choices[0].delta)


# # accumulated_content = ""
# # tool_calls = []
# # tool_ids = set()
# # accumulated_reasoning_content = ""
# # for chunk in response:
# #     if not chunk:
# #         continue

# #     msg: ModelResponseStream = chunk  # type: ignore
# #     if msg is None:
# #         continue
# #     if msg.choices is None or len(msg.choices) == 0:
# #         continue
# #     delta = msg.choices[0].delta
# #     if delta is None:
# #         continue

# #     accumulated_content += delta.get("content", "") or ""
# #     accumulated_reasoning_content += delta.get("reasoning_content", "") or ""
# #     if delta.tool_calls:
# #         for tc in delta.tool_calls:
# #             if not tc:
# #                 continue

# #             if tc.id in tool_ids:
# #                 continue

# #             tool_ids.add(tc.id)
# #             tool_calls.append(tc.model_dump())

# #     # TODO Handle audio, but not now will do later


# # print("----")
# # print(accumulated_content)
# # print("----")
# # print(accumulated_reasoning_content)
# # print("----")
# # print(tool_calls)


# async def main():
#     print(response)
#     print(isawaitable(response))
#     res = response
#     print("Awaited Result")
#     print(isasyncgen(res))
#     print(isgenerator(res))
#     print(isasyncgenfunction(res))
#     print(isgeneratorfunction(res))
#     print(iscoroutinefunction(res))
#     print("__anext__", hasattr(res, "__anext__"))
#     print(hasattr(res, "__aiter__"))
#     print(hasattr(res, "__next__"))
#     print(hasattr(res, "__iter__"))
#     try:
#         async for chunk in res:
#             print(chunk)

#     except Exception:
#         pass

#     try:
#         for chunk in res:
#             print("Synchronous Iteration")
#             print(chunk)
#     except Exception:
#         pass


# import asyncio


# asyncio.run(main())
