from backend.app import app


client = app.test_client()


print("\n========================================")
print("       ANALYZE API TEST")
print("========================================\n")


document_path = "data/test.pdf"


with open(document_path, "rb") as doc_file:
    response = client.post(
        "/api/analyze",
        data={
            "document": (
                doc_file,
                "test.pdf"
            )
        },
        content_type="multipart/form-data"
    )


print("Status code:")
print(response.status_code)


print("\nResponse:")

print(response.json)