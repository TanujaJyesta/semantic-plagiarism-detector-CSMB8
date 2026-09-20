from backend.app import app


client = app.test_client()


history_res = client.get("/api/history")
history_data = history_res.json
if history_data.get("analyses"):
    analysis_id = history_data["analyses"][0]["id"]
else:
    analysis_id = 1


print("\n========================================")
print(f"    HISTORY DETAILS API TEST (ID: {analysis_id})")
print("========================================\n")


response = client.get(
    f"/api/history/{analysis_id}"
)


print("Status code:")
print(response.status_code)


print("\nResponse:")

print(response.json)