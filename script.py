import logging
import requests

# Authorization token
token = """eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9 .eyJzdWIiOiJBdXRoZW50aWNhdGlvbiIsImF1ZCI6ImFjY2VzcyIsImlzcyI6IkFVVEhfU0VSVkVSIiwidXNlciI6NTA1MTY3NDE1NDYwMTgxMTk2OCwidGZwIjoiZTY5MTFhY2I0ZDliM2YxNmViYWJmODAzOTQ2ODMwZGMxNTdmNGY1MWE1ODBjYzJlNTg4ODQxYmE4Y2ZiYzU2MyIsImV4cCI6MTc2MzIyMTkzNCwiaWF0IjoxNzYwNjI5OTM0fQ.-aYdqRs1_dsn9BqOzsf3bfTd_Q1ogGmx08HbxEN8RmS40NDiKbzcYqBu4QV4Mo2tZhMoY2OQqTEO48GpaE_GCg"""


course_id = 322
module_id = 555
resource_ids = [1, 2, 3]

# Base URL
url = f"https://training10xapi.10xscale.ai/course-service/v1/courses/{course_id}/modules/{module_id}/resources"


# Headers
headers = {"Authorization": f"Bearer {token}"}

# For each resource_id, hit the API
for resource_id in resource_ids:
    params = {
        "resource_id": resource_id,
        "limit": 50,
        "offset": 0,
        "sort_field": "modified_at",
        "order_type": "desc",
    }
    response = requests.get(url, params=params, headers=headers, timeout=10)
    logging.info(f"Response for resource_id {resource_id}: {response.json()}")
