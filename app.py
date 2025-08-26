import re
import fitz 
import os
import json
import psycopg2
import bcrypt
import jwt
import requests
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify, send_from_directory , Response , abort
from flask_cors import CORS , cross_origin
from flask_mail import Mail, Message
import google.generativeai as genai
import secrets
import tempfile
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from docx import Document
import firebase_admin
from firebase_admin import credentials as credentials
from firebase_admin import auth as fb_auth
import time
from werkzeug.utils import secure_filename

print("debug token:", secrets.token_hex(16))

load_dotenv()

GEN_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEN_API_KEY:
    print("WARNING: GEMINI_API_KEY not found in environment. Set GEMINI_API_KEY in .env")
else:
    genai.configure(api_key=GEN_API_KEY)

GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
    print("WARNING: GitHub OAuth credentials not found. Set GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET in .env")

app = Flask(
    __name__,
    static_folder='dist',  
    static_url_path=''     
)

# Serve React App - at the very beginning after static folder setup
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    # List of Flask API endpoints that should not serve React
    api_endpoints = [
        'register', 'login', 'google-login', 'github-login', 
        'evaluate', 'user-details', 'resume-history', 'update-name', 
        'upload-profile-image', 'forgot-password-link', 'reset-password',
        'health', 'change-password', 'profile_images'
    ]
    
    # Check if this is an API endpoint
    if any(path.startswith(endpoint) for endpoint in api_endpoints):
        abort(404)
    
    # Check if path is requesting a static file
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    
    # For all other routes, serve index.html (React Router will handle it)
    return send_from_directory(app.static_folder, 'index.html')


if os.getenv("FLASK_ENV") == "production":
    allowed_origins = [
        "https://phonalynx.onrender.com",
        "https://www.phonalynx.onrender.com",
        "http://localhost:5173",  
        "http://localhost:5000",
        "https://phonalynx.onrender.com/auth/github/callback",
        "https://www.phonalynx.onrender.com/auth/github/callback"
    ]
else:
    allowed_origins = [
        "http://localhost:5173", 
        "http://127.0.0.1:5173",  
        "http://localhost:5000"
    ]

CORS(app,
     resources={r"/*": {"origins": allowed_origins}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "Accept"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     expose_headers=["Content-Type", "Authorization"],
     max_age=86400)
DATABASE_URL = os.getenv("DATABASE_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "your-default-jwt-secret")


app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("EMAIL_USER"),
    MAIL_PASSWORD=os.getenv("EMAIL_PASSWORD"),
)
mail = Mail(app)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "profile_images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create if not exists
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def get_db_connection():
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        return conn
    except psycopg2.OperationalError as e:
        print("❌ Database connection error:", str(e))
        return None

def generate_token(email):
    payload = {
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def decode_token(token):
    try:
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
        decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        print("✅ Token decoded:", decoded)
        return decoded.get("email")
    except jwt.ExpiredSignatureError:
        print("❌ Token expired")
    except jwt.InvalidTokenError as e:
        print("❌ Invalid token:", str(e))
    return None

FIREBASE_CRED_PATH = os.getenv("FIREBASE_ADMIN_CRED_PATH")
FIREBASE_CRED_JSON = os.getenv("FIREBASE_ADMIN_CRED")

try:
    if FIREBASE_CRED_PATH and os.path.exists(FIREBASE_CRED_PATH):
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin initialized from file path.")

    elif FIREBASE_CRED_JSON:
        # Load JSON string from env → dict
        cred_dict = json.loads(FIREBASE_CRED_JSON)

        # 🔑 Fix private_key newlines (important!)
        if "private_key" in cred_dict:
            cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")

        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin initialized from JSON env var.")

    else:
        print("⚠️ Firebase Admin not initialized. No credentials found.")

except Exception as e:
    print("❌ Error initializing Firebase Admin:", str(e))

def extract_text_from_file(file_storage):
    filename = file_storage.filename.lower()
    try:
        if filename.endswith(".pdf"):
            file_bytes = file_storage.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            file_storage.stream.seek(0)
            return text.strip()

        elif filename.endswith(".docx"):
            file_storage.stream.seek(0)
            document = Document(file_storage)
            text = "\n".join([para.text for para in document.paragraphs])
            file_storage.stream.seek(0)
            return text.strip()

        elif filename.endswith((".png", ".jpg", ".jpeg")):
            file_storage.stream.seek(0)
            image = Image.open(file_storage.stream)
            text = pytesseract.image_to_string(image)
            file_storage.stream.seek(0)
            return text.strip()

        else:
            file_storage.stream.seek(0)
            try:
                text = file_storage.read().decode(errors="ignore")
                file_storage.stream.seek(0)
                return text.strip()
            except Exception:
                return ""
    except Exception as e:
        print("Error extracting file text:", str(e))
        return ""
    
def extract_links(text):
    links = re.findall(r'(https?://[^\s]+)', text)
    return links if links else ["Not found"]

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]+", text)
    return match.group(0) if match else "Not found"

def extract_phone(text):
    match = re.search(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", text)
    return match.group(0) if match else "Not found"

def call_gemini_for_json(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text if response and hasattr(response, "text") else ""
    except Exception as e:
        print("Gemini API error:", str(e))
        return f"ERROR: {str(e)}"

def extract_resume_via_gemini(resume_text):
    prompt = f"""
You are a resume parser. Extract the following details and return ONLY valid JSON with these keys:
Full Name, Email, Phone, Skills, Education, Projects, Certifications, Work Experience, Objective, Interests, Links

Resume:
\"\"\"{resume_text}\"\"\"

Important:
- Return ONLY a single valid JSON object.
- Use keys exactly as listed above.
- For lists, return arrays.
"""
    raw = call_gemini_for_json(prompt)
    try:
        return json.loads(raw)
    except Exception:
        return {"error": "Invalid JSON from Gemini", "raw": raw}

def generate_detailed_evaluation_via_gemini(resume_text, job_description):
    prompt = f"""
You are an expert HR professional. Provide a resume evaluation with STRICT formatting requirements.

FORMATTING RULES:
- Use EXACTLY this structure with numbered sections
- Start each main section with number and period
- NO extra blank lines between sections
- NO indentation or extra spaces
- Keep text concise and professional
- End with ATS Score as "ATS Score: X/100"

REQUIRED STRUCTURE:

1. Overall Assessment
Brief professional summary of candidate's profile and overall suitability.

2. Strengths
First key strength with specific example
Second strength with relevant details
Third strength highlighting relevant experience

3. Weaknesses
Primary weakness with improvement suggestion
Secondary area needing development
Missing skills or experience gaps

4. Alignment with Job Description
How well resume matches job requirements
Relevant experience alignment
Skills compatibility assessment

5. Skills to Improve & Suggested Courses
Technical skill gaps with course recommendations
Professional development areas
Industry-specific certifications needed

6. Specific Resume Improvements
Content and structure suggestions
Keyword optimization recommendations
Format and presentation enhancements

7. Conclusion
Final recommendation and overall assessment summary.

8. ATS Score: [X/100]

Resume:
\"\"\"{resume_text}\"\"\"

Job Description:
\"\"\"{job_description}\"\"\"

IMPORTANT: Follow the exact format above. No markdown formatting, no extra spaces, no code blocks."
"""
    return call_gemini_for_json(prompt)
def match_with_job(skills, job_desc):
    if not skills:
        return 0
    if isinstance(skills, str):
        skills_list = [s.strip() for s in skills.split(",") if s.strip()]
    elif isinstance(skills, list):
        skills_list = skills
    else:
        try:
            skills_list = list(skills)
        except Exception:
            skills_list = []
    jd_lower = job_desc.lower()
    matched = [s.lower() for s in skills_list if s.lower() in jd_lower]
    return round((len(matched) / max(1, len(skills_list))) * 100, 2)

# Routes
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    if not all([name, email, password]):
        return jsonify({"error": "All fields are required"}), 400

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return jsonify({"error": "User already exists"}), 409

        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                    (name, email, hashed_pw.decode()))
        conn.commit()
        cur.close()
        conn.close()

        token = generate_token(email)
        return jsonify({"message": "Registered successfully", "token": token})
    except Exception as e:
        print("Register error:", str(e))
        return jsonify({"error": "Internal server error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and bcrypt.checkpw(password.encode(), user[0].encode()):
            token = generate_token(email)
            return jsonify({"message": "Login successful", "token": token})
        return jsonify({"error": "Invalid credentials"}), 401
    except Exception as e:
        print("Login error:", str(e))
        return jsonify({"error": "Internal server error"}), 500

@app.route("/google-login", methods=["POST"])
def google_login():
    print("🚀 Google login request received (Firebase-only)")

    try:
        # Force JSON parsing
        data = request.get_json(force=True, silent=False)
        if not data or "token" not in data:
            return jsonify({"error": "No token provided"}), 400

        id_token_str = data["token"]
        print(f"✅ Firebase ID token received, length: {len(id_token_str)}")
        print("🔍 Verifying Firebase token with Admin SDK...")

        # ✅ Verify Firebase ID token
        decoded = fb_auth.verify_id_token(id_token_str)
        email = decoded.get("email")
        name = decoded.get("name") or decoded.get("display_name") or "User"
        picture = decoded.get("picture")

        if not email:
            return jsonify({"error": "No email found in token"}), 400

        # ✅ Create or update user in DB
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database error"}), 500

        try:
            cur = conn.cursor()
            cur.execute("SELECT email FROM users WHERE email=%s", (email,))
            user = cur.fetchone()

            if not user:
                cur.execute(
                    "INSERT INTO users (name, email, password, profile_image) VALUES (%s, %s, %s, %s)",
                    (name, email, "", picture if picture else None)
                )
                conn.commit()
                print(f"✅ User created: {email}")
            else:
                cur.execute(
                    "UPDATE users SET name = COALESCE(%s, name), profile_image = COALESCE(%s, profile_image) WHERE email = %s",
                    (name, picture, email)
                )
                conn.commit()
                print(f"✅ User exists/updated: {email}")
        finally:
            cur.close()
            conn.close()

        # ✅ Issue your app’s JWT
        jwt_token = generate_token(email)
        print("🎉 Google login successful!")

        return jsonify({
            "token": jwt_token,
            "message": "Google login successful"
        }), 200

    except fb_auth.ExpiredIdTokenError:
        return jsonify({"error": "Expired Firebase token"}), 401
    except fb_auth.InvalidIdTokenError:
        return jsonify({"error": "Invalid Firebase token"}), 401
    except Exception as e:
        print(f"❌ Unexpected error in /google-login: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500
    
# Frontend route → let React handle the page
@app.route('/auth/github/callback')
def github_callback_page():
    """Serve React app for GitHub callback page"""
    return send_from_directory(app.static_folder, 'index.html')


# Backend API route → handle GitHub login
@app.route("/api/github-login", methods=["POST"])
def github_login():
    """Handle GitHub OAuth login"""
    data = request.get_json()
    code = data.get("code")

    if not code:
        return jsonify({"error": "Missing GitHub code"}), 400

    try:
        # 1. Exchange code for access token
        headers = {"Accept": "application/json"}
        token_res = requests.post(
            "https://github.com/login/oauth/access_token",
            headers=headers,
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
        )
        token_res.raise_for_status()
        token_json = token_res.json()
        access_token = token_json.get("access_token")

        if not access_token:
            return jsonify({"error": "GitHub login failed"}), 400

        # 2. Fetch user info
        user_res = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {access_token}"}
        )
        user_res.raise_for_status()
        user_data = user_res.json()

        email = user_data.get("email")
        if not email:
            # fallback to user/emails
            emails_res = requests.get(
                "https://api.github.com/user/emails",
                headers={"Authorization": f"token {access_token}"}
            )
            emails_res.raise_for_status()
            emails = emails_res.json()
            primary_email = next((e["email"] for e in emails if e["primary"]), None)
            email = primary_email or (emails[0]["email"] if emails else None)

        if not email:
            return jsonify({"error": "No email found in GitHub account"}), 400

        # 3. Create or update user in DB
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE email=%s", (email,))
        user = cur.fetchone()
        if not user:
            cur.execute(
                "INSERT INTO users (email, provider) VALUES (%s, %s) RETURNING id",
                (email, "github")
            )
            user_id = cur.fetchone()[0]
        else:
            user_id = user[0]
        conn.commit()
        cur.close()
        conn.close()

        # 4. Generate JWT
        token = jwt.encode(
            {"email": email, "exp": datetime.utcnow() + timedelta(hours=24)},
            JWT_SECRET,
            algorithm="HS256"
        )

        return jsonify({"token": token, "email": email}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/forgot-password-link", methods=["POST"])
def send_reset_link():
    data = request.get_json()
    email = data.get("email")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cur.fetchone()

        if not user:
            return jsonify({"error": "No account with that email"}), 404

        reset_token = generate_token(email)
        base_url = os.getenv('BASE_URL')
        reset_link = f"{base_url}/reset-password/{reset_token}"

        msg = Message("Reset Your Password", sender=app.config['MAIL_USERNAME'], recipients=[email])
        msg.body = f"""
Hello,

We received a request to reset your password.
Click the link below to set a new password:
{reset_link}

If you didn't request this, you can ignore this email.

Thanks,
Phonalynx AI
"""
        mail.send(msg)
        return jsonify({"message": "Password reset link sent!"})
    except Exception as e:
        import traceback
        traceback.print_exc()  # Log the error in console
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/reset-password/<token>')
def reset_password_page(token):
    """Serve the React app for password reset page"""
    return send_from_directory(app.static_folder, 'index.html')

# Add API routes that return JSON
@app.route('/api/reset-password/<token>', methods=['GET', 'POST'])
def api_reset_password(token):
    """API endpoint for token validation and password reset"""
    
    if request.method == 'GET':
        # Validate token
        email = decode_token(token)
        if not email:
            return jsonify({"error": "Invalid or expired token"}), 400
        return jsonify({"message": "Valid token"}), 200
    
    elif request.method == 'POST':
        # Reset password
        data = request.get_json()
        new_password = data.get("new_password")
        
        if not new_password:
            return jsonify({"error": "New password is required"}), 400
        
        email = decode_token(token)
        if not email:
            return jsonify({"error": "Invalid or expired token"}), 400
        
        hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("UPDATE users SET password=%s WHERE email=%s", (hashed_pw, email))
            conn.commit()
            cur.close()
            conn.close()
            return jsonify({"message": "Password reset successful!"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
@app.route("/evaluate", methods=["POST"])
def evaluate_resume():
    token = request.headers.get("Authorization")
    email = None

    if token:
        email = decode_token(token)
        if not email:
            email = None

    if 'resume' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Resume file and job description are required"}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    try:
        resume_text = extract_text_from_file(resume_file)

        parsed_json = extract_resume_via_gemini(resume_text)

        evaluation = generate_detailed_evaluation_via_gemini(resume_text, job_description)

        match = re.search(r'ATS Score\s*\(.*?\):\s*(\d+)', evaluation)
        score = int(match.group(1)) if match else 0

        if email:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO resume_history (email, file_name, created_at)
                VALUES (%s, %s, %s)
            """, (email, resume_file.filename, datetime.utcnow()))

            cur.execute("SELECT resumes_analyzed, average_score FROM users WHERE email = %s", (email,))
            user_data = cur.fetchone()

            if user_data:
                old_count = user_data[0] or 0
                old_avg = user_data[1] or 0
                new_count = old_count + 1
                new_avg = round((old_avg * old_count + score) / new_count, 2)

                cur.execute("""
                    UPDATE users SET resumes_analyzed = %s, average_score = %s
                    WHERE email = %s
                """, (new_count, new_avg, email))

            conn.commit()
            cur.close()
            conn.close()

        return jsonify({"evaluation": evaluation, "parsed_resume": parsed_json})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500




@app.route("/resume-history", methods=["GET"])
def resume_history():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    email = decode_token(token)
    if not email:
        return jsonify({"error": "Invalid or expired token"}), 401

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT file_name, created_at FROM resume_history WHERE email = %s", (email,))
        rows = cur.fetchall()
        cur.close()
        conn.close()

        resume_list = [
            {
                "filename": row[0],
                "date": row[1].strftime('%Y-%m-%d %H:%M:%S')
            }
            for row in rows
        ]
        return jsonify(resume_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-name", methods=["POST"])
def update_name():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    email = decode_token(token)
    if not email:
        return jsonify({"error": "Invalid or expired token"}), 401

    data = request.get_json()
    new_name = data.get("name")

    if not new_name:
        return jsonify({"error": "Name is required"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE users SET name=%s WHERE email=%s", (new_name, email))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"message": "Name updated successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/change-password", methods=["POST"])
def change_password():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    email = decode_token(token)
    if not email:
        return jsonify({"error": "Invalid or expired token"}), 401

    data = request.get_json()
    new_password = data.get("password")

    if not new_password:
        return jsonify({"error": "New password is required"}), 400
    
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    try:
        hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE users SET password=%s WHERE email=%s", (hashed_pw, email))
        conn.commit()
        cur.close()
        conn.close()

        return jsonify({"message": "Password changed successfully"})
    except Exception as e:
        print("Change password error:", str(e))
        return jsonify({"error": "Internal server error"}), 500


@app.route("/upload-profile-image", methods=["POST"])
def upload_profile_image():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    email = decode_token(token)
    if not email:
        return jsonify({"error": "Invalid or expired token"}), 401

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    
    # Check file type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image.filename or \
       image.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG, GIF allowed"}), 400

    # Check file size (5MB max)
    if image.content_length and image.content_length > 5 * 1024 * 1024:
        return jsonify({"error": "Image size must be less than 5MB"}), 400

    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cur = conn.cursor()
        
        # Get current profile image to delete old one
        cur.execute("SELECT profile_image FROM users WHERE email=%s", (email,))
        old_image_result = cur.fetchone()
        old_image_path = old_image_result[0] if old_image_result else None
        
        # Delete old profile image if exists and is not an external URL
        if old_image_path and old_image_path.startswith('/profile_images/'):
            old_filename = old_image_path.replace('/profile_images/', '')
            old_filepath = os.path.join(UPLOAD_FOLDER, old_filename)
            if os.path.exists(old_filepath):
                try:
                    os.remove(old_filepath)
                    print(f"Deleted old image: {old_filepath}")
                except Exception as e:
                    print(f"Failed to delete old image: {e}")

        # Generate unique filename with timestamp
        timestamp = int(time.time())
        file_extension = image.filename.rsplit('.', 1)[1].lower()
        safe_email = email.replace('@', '_').replace('.', '_')
        filename = f"profile_{safe_email}_{timestamp}.{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Save image
        image.save(filepath)
        print(f"Saved new image: {filepath}")

        # Verify file was saved correctly
        if not os.path.exists(filepath):
            return jsonify({"error": "Failed to save image"}), 500

        # Store path in DB (consistent format)
        db_path = f"/profile_images/{filename}"

        # Update user's profile image in database
        cur.execute("UPDATE users SET profile_image=%s WHERE email=%s", (db_path, email))
        conn.commit()
        
        # Verify the update was successful
        cur.execute("SELECT profile_image FROM users WHERE email=%s", (email,))
        updated_result = cur.fetchone()
        
        if not updated_result or updated_result[0] != db_path:
            cur.close()
            conn.close()
            return jsonify({"error": "Failed to update database"}), 500

        cur.close()
        conn.close()

        return jsonify({
            "message": "Profile image updated successfully", 
            "profile_image": db_path,
            "timestamp": timestamp
        })
    
    except Exception as e:
        print(f"Profile image upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500


@app.route("/profile_images/<filename>")
def get_profile_image(filename):
    try:
        # Security check - prevent directory traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        # Check if file exists
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(filepath):
            return jsonify({"error": "Image not found"}), 404
            
        response = send_from_directory(app.config["UPLOAD_FOLDER"], filename)
        
        # Add proper headers to prevent aggressive caching but allow some caching
        response.headers['Cache-Control'] = 'public, max-age=300'  # Cache for 5 minutes
        response.headers['ETag'] = f'"{filename}-{os.path.getmtime(filepath)}"'
        
        return response
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return jsonify({"error": "Failed to load image"}), 500


@app.route("/user-details", methods=["GET"])
def user_details():
    token = request.headers.get("Authorization")
    if not token:
        return jsonify({"error": "Unauthorized"}), 401

    email = decode_token(token)
    if not email:
        return jsonify({"error": "Invalid or expired token"}), 401

    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500
            
        cur = conn.cursor()
        cur.execute("""
            SELECT name, email, profile_image, resumes_analyzed, average_score
            FROM users WHERE email = %s
        """, (email,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if not user:
            return jsonify({"error": "User not found"}), 404

        # Ensure consistent data format
        user_data = {
            "name": user[0] or "",
            "email": user[1] or "",
            "profile_image": user[2] or "",
            "resumes_analyzed": user[3] or 0,
            "average_score": user[4] or 0
        }
        
        print(f"Sending user data: {user_data}")  # Debug log
        
        return jsonify(user_data)
        
    except Exception as e:
        print(f"Error fetching user details: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()}), 200

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Route not found"}), 404


    
if __name__ == '__main__':
    if os.getenv('FLASK_ENV') == 'production':
        from waitress import serve
        print("🚀 Starting production server...")
        serve(app, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
    else:
        print("🔧 Starting development server...")
        app.run(debug=True, port=5000)