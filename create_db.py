from utils.db import get_db_connection
from models.user import create_user_table
from models.order import create_order_table  # Only if you're using an order model
from models.admin import create_admin_table, insert_admin
from models.bottle import create_bottle_table
from models.delivery_person import create_delivery_person_table,insert_delivery_person
conn = get_db_connection()

# Create user and order tables
create_user_table(conn)
create_order_table(conn)  # Skip if not needed or not created yet

# ✅ Create admin table
create_admin_table(conn)

create_bottle_table(conn)

create_delivery_person_table(conn)

# ✅ Create default admin account (only once)
try:
    insert_admin(conn, 'admin', 'admin123')  # username: admin, password: admin123
except:
    pass  # If already exists, skip


# ✅ Insert delivery persons (only once)
try:
    insert_delivery_person(conn, 'umer', '03001234567')
    insert_delivery_person(conn, 'ahmed', '03111234567')
    insert_delivery_person(conn, 'qasim', '03211234567')
except:
    pass  # Prevents crash if already inserted


print("Database and tables created successfully!")

