from flask import Blueprint, render_template, request, redirect, session, flash
from utils.db import get_db_connection
from models.user import insert_user, find_user_by_email

customer_bp = Blueprint('customer', __name__)

@customer_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        existing_user = find_user_by_email(conn, email)
        if existing_user:
            flash('Email already registered!', 'error')
            return redirect('/customer/register')
        insert_user(conn, name, email, password)
        flash('Registered successfully! Please login.', 'success')
        return redirect('/customer/login')
    return render_template('register.html')

@customer_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = get_db_connection()
        user = find_user_by_email(conn, email)
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash('Login successful!', 'success')
            return 'Welcome to Dashboard (Coming Soon)'
        else:
            flash('Invalid credentials!', 'error')
            return redirect('/customer/login')
    return render_template('login.html')
