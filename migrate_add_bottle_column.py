import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

try:
    cursor.execute("ALTER TABLE orders ADD COLUMN bottle_type TEXT")
    print("✅ 'bottle_type' column added to orders table.")
except sqlite3.OperationalError as e:
    print("⚠️ Column already exists or error occurred:", e)

try:
    cursor.execute("ALTER TABLE orders ADD COLUMN created_at TEXT")
    print("✅ 'created_at' column added to orders table.")
except sqlite3.OperationalError as e:
    print("⚠️ Column already exists or error occurred:", e)
    # column might already exist, so just pass
    

try:
    conn.execute("ALTER TABLE orders ADD COLUMN delivery_person_id INTEGER")
    print("Column added.")
except sqlite3.OperationalError as e:

    print("Column might already exist or error:", e)


    try:
        conn.execute("ALTER TABLE delivery_persons ADD COLUMN username TEXT UNIQUE")
        print("Column added")
    except sqlite3.OperationalError as e:
        print("✅ 'username' column already exists")

    # Add password column if it doesn't exist
    try:
        conn.execute("ALTER TABLE delivery_persons ADD COLUMN password TEXT")
        print("Column added")
    except sqlite3.OperationalError as e:
        print("✅ 'password' column already exists")


conn.commit()
conn.close()
