from flask import Flask
from routes.customer_routes import customer_bp
from routes.order_routes import order_bp
from routes.admin_routes import admin_bp
from routes.bottle_routes import bottle_bp
from routes.delivery_routes import delivery_bp
from flask import Flask, render_template, request, redirect, session, flash
from utils.db import get_db_connection
from models.user import verify_user
from models.admin import verify_admin
from models.delivery_person import verify_delivery_person
from routes.customer_routes import customer_bp
from routes.admin_routes import admin_bp
from routes.delivery_routes import delivery_bp
from routes.customer_routes import customer_bp
from routes.admin_routes import admin_bp
from routes.delivery_routes import delivery_bp
from utils.db import get_db_connection
from routes.payment_routes import payment_bp


app = Flask(__name__)
app.secret_key = 'your_secret_key'  # required for session

# Register Blueprints

app.register_blueprint(order_bp, url_prefix='/orders')
app.register_blueprint(admin_bp, url_prefix='/admin')
app.register_blueprint(bottle_bp, url_prefix='/bottles')
app.register_blueprint(delivery_bp, url_prefix='/delivery')
app.register_blueprint(customer_bp, url_prefix='/customer')
app.register_blueprint(payment_bp)







@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'] 
        password = request.form['password']
        role = request.form['role']

        conn = get_db_connection()

        if role == 'customer':
            user = verify_user(conn, username, password)
            if user:
                session['user_id'] = user['id']
                session['user_name'] = user['name']
                session['role'] = 'customer'
                return redirect('/orders/place')
            else:
                flash('Invalid customer credentials!')

        elif role == 'admin':
            user = verify_admin(conn, username, password)
            if user:
                session['admin_id'] = user['id']
                session['role'] = 'admin'
                return redirect('/admin/dashboard')
            else:
                flash('Invalid admin credentials!')

        elif role == 'delivery':
            user = verify_delivery_person(conn, username, password)
            if user:
                session['delivery_id'] = user['id']
                session['role'] = 'delivery'
                return redirect('/delivery/dashboard')
            else:
                flash('Invalid delivery person credentials!')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')



if __name__ == '__main__':
    app.run(debug=True)