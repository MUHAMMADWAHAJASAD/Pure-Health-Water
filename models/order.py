import datetime
from utils.db import get_db_connection

def create_order_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            bottle_type TEXT,
            bottle_qty INTEGER,
            delivery_address TEXT,
            status TEXT DEFAULT 'Pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            delivery_person_id INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(delivery_person_id) REFERENCES delivery_persons(id)
        )
    ''')
    conn.commit()

def insert_order(conn, user_id, bottle_type, bottle_qty, delivery_address, payment_mode):
    cursor = conn.cursor()
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO orders (
            user_id, bottle_type, bottle_qty, delivery_address,
            created_at, payment_mode, payment_status
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, bottle_type, bottle_qty, delivery_address, created_at, payment_mode, 'Pending'))
    conn.commit()

def get_orders_by_user(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT o.id, o.bottle_type, o.bottle_qty, o.delivery_address, o.status, o.created_at,
               o.payment_mode, o.payment_status,
               dp.name AS delivery_person_name, dp.phone AS delivery_person_phone
        FROM orders o
        LEFT JOIN delivery_persons dp ON o.delivery_person_id = dp.id
        WHERE o.user_id = ?
        ORDER BY o.created_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    orders = []
    for row in rows:
        orders.append({
            'id': row[0],
            'bottle_type': row[1],
            'bottle_qty': row[2],
            'delivery_address': row[3],
            'status': row[4],
            'created_at': row[5],
            'payment_mode': row[6],
            'payment_status': row[7],
            'delivery_person_name': row[8],
            'delivery_person_phone': row[9]
        })
    return orders

def get_all_orders(conn):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT orders.id, users.email, orders.bottle_type, orders.bottle_qty,
               orders.delivery_address, orders.status, orders.created_at,
               orders.payment_mode, orders.payment_status,
               dp.name as delivery_person
        FROM orders
        JOIN users ON orders.user_id = users.id
        LEFT JOIN delivery_persons dp ON orders.delivery_person_id = dp.id
        ORDER BY orders.created_at DESC
    """)
    rows = cursor.fetchall()
    orders = []
    for row in rows:
        orders.append({
            'id': row[0],
            'username': row[1],
            'bottle_type': row[2],
            'bottle_qty': row[3],
            'delivery_address': row[4],
            'status': row[5],
            'created_at': row[6],
            'payment_mode': row[7],
            'payment_status': row[8],
            'delivery_person': row[9] or 'Unassigned'
        })
    return orders

def assign_delivery_person(conn, order_id, delivery_person_id):
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE orders SET delivery_person_id = ? WHERE id = ?
    """, (delivery_person_id, order_id))
    conn.commit()