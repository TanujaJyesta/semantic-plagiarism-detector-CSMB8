from backend.database.database import engine, Base
from backend.database import models


print("\nCreating database tables...\n")


Base.metadata.create_all(
    bind=engine
)


print("Database created successfully!")
print("SQLite database: plagiarism.db")