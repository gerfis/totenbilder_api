import os
import uuid
from typing import Dict
from fastapi import APIRouter, HTTPException, Response, Request, status, Depends
from pydantic import BaseModel
import bcrypt

import mysql.connector
from dotenv import load_dotenv

load_dotenv()

# Database Config (Redundant but safe)
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

router = APIRouter()



# In-memory session store (Note: cleared on restart, not shared between workers)
SESSIONS: Dict[str, str] = {}

class LoginRequest(BaseModel):
    username: str
    password: str

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error connecting to database: {err}")
        return None

def verify_password(plain_password, hashed_password):
    try:
        # bcrypt.checkpw expects bytes
        # hashed_password from DB is string, possibly with leading $2b$ or similar
        # plain_password from user is string
        if not hashed_password:
            return False
            
        # Ensure correct encoding (utf-8)
        pwd_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8')
        
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception as e:
        print(f"Password verify error: {e}")
        return False

@router.post("/login")
async def login(response: Response, login_data: LoginRequest):
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=500, detail="Database connection error")
    
    try:
        cursor = conn.cursor(dictionary=True)
        # Check against 'users' table: name, pass
        query = "SELECT name, pass FROM users WHERE name = %s"
        cursor.execute(query, (login_data.username,))
        user = cursor.fetchone()
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
        if not verify_password(login_data.password, user['pass']):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create Session
        token = str(uuid.uuid4())
        SESSIONS[token] = user['name']
        
        # Set Cookie
        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=False # Set True in production with HTTPS
        )
        
        return {"message": "Login successful"}
        
    finally:
        if conn:
            conn.close()

@router.post("/logout")
async def logout(response: Response, request: Request):
    token = request.cookies.get("session_token")
    if token and token in SESSIONS:
        del SESSIONS[token]
    
    response.delete_cookie("session_token")
    return {"message": "Logged out"}

def get_current_user_from_token(token: str):
    return SESSIONS.get(token)
