from flask import Blueprint, request, render_template, redirect, session, flash
from utils.db import get_db_connection

payment_bp = Blueprint('payment', __name__)

@payment_bp.route('/payment/easypaisa/<int:order_id>', methods=['GET', 'POST'])
def easypaisa_payment(order_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Fetch order by ID
    cursor.execute('SELECT * FROM orders WHERE id = ?', (order_id,))
    order = cursor.fetchone()

    if not order:
        flash('❌ Order not found.', 'error')
        return redirect('/orders/my')

    # Fetch bottle price based on bottle_type
    cursor.execute('SELECT price FROM bottles WHERE bottle_type = ?', (order['bottle_type'],))
    bottle = cursor.fetchone()

    if not bottle:
        flash('❌ Bottle type not found.', 'error')
        return redirect('/orders/my')

    total_price = bottle['price'] * order['bottle_qty']

    # On POST, process mock Easypaisa payment
    if request.method == 'POST':
        phone = request.form['phone']
        code = request.form['code']

        if code == '123456':  # Mock success
            cursor.execute('UPDATE orders SET payment_status = ?, payment_mode = ? WHERE id = ?', 
                           ('Confirmed', 'Easypaisa', order_id))
            conn.commit()
            flash('✅ Payment Successful & Order Confirmed!', 'success')
            return redirect('/orders/orders/my')
        else:
            flash('❌ Invalid Easypaisa Code. Try again.', 'error')

    conn.close()
    return render_template('easypaisa_payment.html', order=order, total_price=total_price)