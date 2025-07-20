import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()


try:
    cursor.execute("ALTER TABLE orders ADD COLUMN payment_mode TEXT;")
    print("✅ 'payment type' column added to orders table.")
except sqlite3.OperationalError as e:
    print("⚠️ Column already exists or error occurred:", e)


try:
    cursor.execute("ALTER TABLE orders ADD COLUMN payment_status TEXT DEFAULT 'Pending';")
    print("✅ 'payment status' column added to orders table.")
except sqlite3.OperationalError as e:
    print("⚠️ Column already exists or error occurred:", e)
