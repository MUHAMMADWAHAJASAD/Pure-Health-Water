from flask import Blueprint, render_template, request, redirect, session, flash
from utils.db import get_db_connection
from models.admin import verify_admin, get_all_orders, update_order_status
from models.delivery_person import get_all_delivery_persons  # ✅ new import
from models.order import assign_delivery_person  # ✅ new import

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/login', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        admin = verify_admin(conn, username, password)
        if admin:
            session['admin'] = admin['username']
            return redirect('/admin/dashboard')
        else:
            flash('Invalid admin credentials!', 'error')
    return render_template('admin_login.html')

@admin_bp.route('/dashboard')
def admin_dashboard():
    if 'admin' not in session:
        return redirect('/admin/login')
    conn = get_db_connection()
    orders = get_all_orders(conn)
    delivery_persons = get_all_delivery_persons(conn)  # ✅ fetch delivery persons
    return render_template('admin_dashboard.html', orders=orders, delivery_persons=delivery_persons)

@admin_bp.route('/update_status/<int:order_id>/<string:new_status>')
def update_status(order_id, new_status):
    if 'admin' not in session:
        return redirect('/admin/login')
    conn = get_db_connection()
    update_order_status(conn, order_id, new_status)
    flash('Order status updated!')
    return redirect('/admin/dashboard')

@admin_bp.route('/assign_delivery/<int:order_id>', methods=['POST'])
def assign_delivery(order_id):
    if 'admin' not in session:
        return redirect('/admin/login')
    delivery_person_id = int(request.form['delivery_person_id'])
    conn = get_db_connection()
    assign_delivery_person(conn, order_id, delivery_person_id)
    flash('Delivery person assigned!','message')
    return redirect('/admin/dashboard')

