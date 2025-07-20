def create_admin_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()

def insert_admin(conn, username, password):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO admins (username, password) VALUES (?, ?)', (username, password))
    conn.commit()

def verify_admin(conn, username, password):
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM admins WHERE username = ? AND password = ?', (username, password))
    return cursor.fetchone()

def get_all_orders(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT orders.id, users.name, orders.bottle_qty, orders.delivery_address, 
               orders.status, orders.created_at, orders.delivery_person_id,
               delivery_persons.name AS delivery_person_name
        FROM orders
        JOIN users ON orders.user_id = users.id
        LEFT JOIN delivery_persons ON orders.delivery_person_id = delivery_persons.id
        ORDER BY orders.created_at DESC
    """)
    rows = cursor.fetchall()
    orders = []
    for row in rows:
        orders.append({
            'id': row[0],
            'username': row[1],
            'bottle_qty': row[2],
            'delivery_address': row[3],
            'status': row[4],
            'created_at': row[5],
            'delivery_person_id': row[6],
            'delivery_person_name': row[7] or "Not Assigned"
        })
    return orders


def update_order_status(conn, order_id, new_status):
    cursor = conn.cursor()
    cursor.execute('UPDATE orders SET status = ? WHERE id = ?', (new_status, order_id))
    conn.commit()
