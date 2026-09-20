from backend.app import app


client = app.test_client()


print("\n========================================")
print("       HISTORY API TEST")
print("========================================\n")


response = client.get("/api/history")


print("Status code:")
print(response.status_code)


print("\nResponse:")

print(response.json)