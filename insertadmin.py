from utils.db import get_db_connection
get_db_connection
from models.admin import insert_admin
conn = get_db_connection()

try:
    insert_admin(conn, 'admin12@gmail.com', 'admin000')  # username: admin, password: admin123
except:
    pass  
