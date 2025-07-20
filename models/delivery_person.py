import hashlib


def create_delivery_person_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS delivery_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL
        )
    ''')
    conn.commit()

def insert_delivery_person(conn, name, phone, username, password):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO delivery_persons (name, phone, username, password) VALUES (?, ?, ?, ?)",
        (name, phone, username, password)
    )
    conn.commit()

def verify_delivery_person(conn, username, password):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM delivery_persons WHERE username = ? AND password = ?", (username, password))
    row = cursor.fetchone()
    if row:
        return dict(row)
    return None

def get_all_delivery_persons(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM delivery_persons")
    return cursor.fetchall()

def get_assigned_orders(conn, delivery_person_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT orders.*, users.name 
        FROM orders 
        JOIN users ON orders.user_id = users.id 
        WHERE orders.delivery_person_id = ?
    """, (delivery_person_id,))
    return cursor.fetchall()

def update_order_status_by_delivery(conn, order_id, status):
    cursor = conn.cursor()
    cursor.execute("UPDATE orders SET status = ? WHERE id = ?", (status, order_id))
    conn.commit()