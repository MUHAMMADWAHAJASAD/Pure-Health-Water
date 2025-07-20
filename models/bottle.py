def create_bottle_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bottles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            size_liters INTEGER NOT NULL,
            price REAL NOT NULL
        )
    ''')
    conn.commit()

def insert_bottle(conn, name, size_liters, price):
    cursor = conn.cursor()
    cursor.execute('INSERT INTO bottles (size_liters, name, price) VALUES (?, ?, ?)', ( size_liters, name, price))
    conn.commit()

def get_all_bottles(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, size_liters, price FROM bottles')
    rows = cursor.fetchall()
    return [{'id': row[0], 'name': row[1], 'size_liters': row[2], 'price': row[3]} for row in rows]


def delete_bottle(conn, bottle_id):
    cursor = conn.cursor()
    cursor.execute('DELETE FROM bottles WHERE id = ?', (bottle_id,))
    conn.commit()
