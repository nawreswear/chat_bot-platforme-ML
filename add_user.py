from werkzeug.security import generate_password_hash
import sqlite3

username = 'test_user'
password = 'testpassword'
role = 'user'

hashed_password = generate_password_hash(password)

conn = sqlite3.connect('users.db')
cursor = conn.cursor()
cursor.execute('INSERT INTO users (username, password, role) VALUES (?, ?, ?)',
               (username, hashed_password, role))
conn.commit()
conn.close()

print(f"User {username} added successfully.")