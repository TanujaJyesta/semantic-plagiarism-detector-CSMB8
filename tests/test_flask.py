from backend.app import app


client = app.test_client()


print("\n========================================")
print("        FLASK API TEST")
print("========================================\n")


# Test home endpoint
response = client.get("/")

print("Home endpoint:")
print("Status code:", response.status_code)
print("Response:", response.json)


# Test health endpoint
response = client.get("/api/health")

print("\nHealth endpoint:")
print("Status code:", response.status_code)
print("Response:", response.json)