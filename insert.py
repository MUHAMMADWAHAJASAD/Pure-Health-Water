from utils.db import get_db_connection
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("INSERT INTO delivery_persons (name, phone, username, password) VALUES (?, ?, ?, ?)",
               ("Ali Raza", "03123456789", "ali123", "pass123"))
conn.commit()
conn.close()
print("Dummy delivery person added!")