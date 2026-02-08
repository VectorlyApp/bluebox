import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

headers = {
    "Content-Type": "application/json",
    "X-Service-Token": os.getenv("VECTORLY_API_KEY")
}

print(headers)


# 2. Get Public Routines
url_public = "https://api.sandbox.vectorly.app/api/v1/routines/public"
response = requests.get(url_public, headers=headers)
data = response.json()
print(data[0][
])
print("\nPublic Routines:")
# print(json.dumps(data, indent=2))
