import sqlite3

def add_missing_columns():
    conn = sqlite3.connect('database.db')  # Adjust path if needed
    cursor = conn.cursor()

    # Fetch table schema
    cursor.execute("PRAGMA table_info(delivery_persons)")
    columns = [column[1] for column in cursor.fetchall()]

    if 'username' not in columns:
        try:
            cursor.execute("ALTER TABLE delivery_persons ADD COLUMN username TEXT")
            print("✅ 'username' column added.")
        except sqlite3.OperationalError as e:
            print("❌ Error adding 'username' column:", e)
    else:
        print("✅ 'username' column already exists.")

    if 'password' not in columns:
        try:
            cursor.execute("ALTER TABLE delivery_persons ADD COLUMN password TEXT")
            print("✅ 'password' column added.")
        except sqlite3.OperationalError as e:
            print("❌ Error adding 'password' column:", e)
    else:
        print("✅ 'password' column already exists.")

    conn.commit()
    conn.close()

# Call the function
add_missing_columns()
