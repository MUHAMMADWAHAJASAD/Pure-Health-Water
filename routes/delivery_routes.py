from flask import Blueprint, render_template, request, redirect, session, flash
from utils.db import get_db_connection
from models.delivery_person import verify_delivery_person, get_assigned_orders, update_order_status_by_delivery

delivery_bp = Blueprint('delivery', __name__)

@delivery_bp.route('/login', methods=['GET', 'POST'])
def login_delivery():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        delivery_person = verify_delivery_person(conn, username, password)
        if delivery_person:
            session['delivery_person'] = delivery_person
            return redirect('/delivery/dashboard')
        else:
            flash('Invalid credentials!', 'error')
    return render_template('delivery_login.html')

@delivery_bp.route('/dashboard')
def delivery_dashboard():
    if 'delivery_person' not in session:
        return redirect('/delivery/login')
    conn = get_db_connection()
    orders = get_assigned_orders(conn, session['delivery_person']['id'])
    return render_template('delivery_dashboard.html', orders=orders, delivery_person=session['delivery_person'])

@delivery_bp.route('/update_status/<int:order_id>/<string:new_status>')
def update_order_status(order_id, new_status):
    if 'delivery_person' not in session:
        return redirect('/delivery/login')
    conn = get_db_connection()
    update_order_status_by_delivery(conn, order_id, new_status)
    flash('Order status updated!')
    return redirect('/delivery/dashboard')
