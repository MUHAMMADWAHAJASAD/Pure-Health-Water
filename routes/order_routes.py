from flask import Blueprint, render_template, request, redirect, session, flash, url_for
from utils.db import get_db_connection
from models.order import insert_order, get_orders_by_user
from models.bottle import get_all_bottles  # Import bottle helper

order_bp = Blueprint('order', __name__)

@order_bp.route('/place', methods=['GET', 'POST'])
def place_order():
    if 'user_id' not in session:
        flash('Please login first.', 'error')
        return redirect('/customer/login')
    
    conn = get_db_connection()
    bottles = get_all_bottles(conn)

    if request.method == 'POST':
        bottle_type = request.form['bottle_type']
        bottle_qty = int(request.form['bottle_qty'])
        delivery_address = request.form['delivery_address']
        payment_method = request.form['payment_mode']

        # If payment method is COD, insert order directly
        if payment_method == 'cod':
            insert_order(conn, session['user_id'], bottle_type, bottle_qty, delivery_address, 'cod')
            flash('Order placed successfully via Cash on Delivery!', 'success')
            return redirect('/orders/orders/my')
        
        elif payment_method == 'easypaisa':
            # Store temporary session data for Easypaisa redirection
            session['pending_order'] = {
                'bottle_type': bottle_type,
                'bottle_qty': bottle_qty,
                'delivery_address': delivery_address
            }
            return redirect(url_for('order.easypaisa_payment'))

    return render_template('place_order.html', bottles=bottles)

@order_bp.route('/orders/my')
def my_orders():
    if 'user_id' not in session:
        flash('Please login first.', 'error')
        return redirect('/customer/login')
    
    conn = get_db_connection()
    orders = get_orders_by_user(conn, session['user_id'])
    return render_template('my_orders.html', orders=orders)

# --- Easypaisa Mock Payment Flow ---
@order_bp.route('/payment/easypaisa', methods=['GET', 'POST'])
def easypaisa_payment():
    if 'pending_order' not in session or 'user_id' not in session:
        flash('No pending order found. Please place an order first.', 'error')
        return redirect('/place')

    order_data = session['pending_order']

    # Get price of selected bottle from DB
    conn = get_db_connection()
    cursor = conn.execute('SELECT price FROM bottles WHERE id = ?', (order_data['bottle_type'],))
    bottle = cursor.fetchone()

    if not bottle:
        flash('Invalid bottle type selected.', 'error')
        return redirect('/place')

    total_price = bottle['price'] * int(order_data['bottle_qty'])

    if request.method == 'POST':
        # In real case, validate Easypaisa payment
        insert_order(conn, session['user_id'],
                     order_data['bottle_type'],
                     order_data['bottle_qty'],
                     order_data['delivery_address'],
                     'easypaisa')
        session.pop('pending_order')
        flash('Order placed successfully via Easypaisa!', 'success')
        return redirect('/orders/orders/my')

    return render_template('easypaisa_payment.html',
                           order=order_data,
                           total_price=total_price)