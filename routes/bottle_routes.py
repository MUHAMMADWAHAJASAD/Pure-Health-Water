from flask import Blueprint, render_template, request, redirect, flash, session
from utils.db import get_db_connection
from models.bottle import insert_bottle, get_all_bottles, delete_bottle

bottle_bp = Blueprint('bottle', __name__)

@bottle_bp.route('/list', methods=['GET', 'POST'])
def bottle_list():
    if 'admin' not in session:
        return redirect('/admin/login')

    conn = get_db_connection()

    if request.method == 'POST':
        
        name = request.form['name']
        size = int(request.form['size_liters'])
        price = float(request.form['price'])
        insert_bottle(conn,name, size, price)
        flash('Bottle added!')

    bottles = get_all_bottles(conn)
    return render_template('bottle_list.html', bottles=bottles)

@bottle_bp.route('/delete/<int:bottle_id>')
def delete_bottle_route(bottle_id):
    if 'admin' not in session:
        return redirect('/admin/login')

    conn = get_db_connection()
    delete_bottle(conn, bottle_id)
    flash('Bottle deleted!')
    return redirect('/bottles/list')
