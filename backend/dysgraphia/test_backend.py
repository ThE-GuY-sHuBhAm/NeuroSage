import requests

BASE_URL = "http://127.0.0.1:5000"

# 1. Test Questions Endpoint
print("Testing /api/generate-questions...")
q_response = requests.post(f"{BASE_URL}/api/generate-questions", json={"age": 8})
print("Status:", q_response.status_code)
print("Response:", q_response.json())
print("-" * 50)

# 2. Test Analysis Endpoint (High Concern)
print("Testing /api/generate-analysis (High Concern)...")

high_concern_payload = {
    "userInfo": {"name": "Alex", "age": 8},
    "questionnaireScore": 45,
    "imageScore": 85.0,
    "finalScore": 85.5,
    "questions": [
        {"category": "motor_skills", "question": "Does their writing hand cramp up?"},
        {"category": "letter_formation", "question": "Do they consistently fail to close letters?"}
    ],
    "answers": {
        "0": 4, # 4 = Always
        "1": 3  # 3 = Frequently
    }
}

a_response = requests.post(f"{BASE_URL}/api/generate-analysis", json=high_concern_payload)
print("Status:", a_response.status_code)

# Let's print just the analysis text so it's easy to read in the terminal
response_json = a_response.json()
print("\n--- GENERATED REPORT ---")
print(response_json.get('analysis', 'No analysis returned'))
print("------------------------\n")