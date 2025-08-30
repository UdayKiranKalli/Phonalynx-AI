# 📱 Phonalynx

[![Deploy on Render](https://img.shields.io/badge/Deployed%20on-Render-blue?logo=render)](https://phonalynx.onrender.com/)
[![Made with Flask](https://img.shields.io/badge/Made%20with-Flask-green?logo=flask)](https://flask.palletsprojects.com/)
[![Made with React](https://img.shields.io/badge/Made%20with-React-61DAFB?logo=react)](https://react.dev/)
[![Dockerized](https://img.shields.io/badge/Containerized-Docker-blue?logo=docker)](https://www.docker.com/)
[![GitHub stars](https://img.shields.io/github/stars/UdayKiranKalli/Phonalynx?style=social)](https://github.com/UdayKiranKalli/Phonalynx/stargazers)

---

## 🚀 Live Demo
👉 [Phonalynx on Render](https://phonalynx.onrender.com/)

---

📌 About the Project
## 📌 About the Project
**Phonalynx** is a full-stack web application built with **Flask (Python backend)** and **React (frontend)**, containerized with **Docker** and deployed on **Render**.  

It demonstrates:
- 🔐 Authentication (JWT-based + Google & GitHub login)  
- 👤 User profile management (store & display user details)  
- 📂 Resume evaluation & parsing  
- 🐳 Dockerized deployment  
- 🎨 React UI served through Flask (`dist/`)  

✨ Features
## ✨ Features
- User Authentication  
  - Email/Password login  
  - Google OAuth login  
  - GitHub OAuth login  
- User Profile Management (name, email, avatar, etc.)  
- Resume Upload & Parsing (PDF/DOCX support)  
- Secure API with Flask & JWT  
- PostgreSQL integration  
- React build (`dist/`) served from Flask  
- Dockerized for easy deployment  
- Live production deployment on Render 

---

## 🛠️ Tech Stack
**Frontend:** React (Vite) + TailwindCSS (built into `dist/`)  
**Backend:** Flask (Python)  
**Database:** PostgreSQL  
**Deployment:** Docker + Render  
**Other Tools:** JWT, bcrypt, PyMuPDF, python-docx  

---

## 📷 Screenshots / Demo
![Demo Screenshot](dist/assets/demo.mp4)  
 

---

## ⚡ Getting Started (Run Locally)

### Prerequisites
- [Python 3.9+](https://www.python.org/downloads/)
- [Node.js](https://nodejs.org/en/download/) (only if rebuilding frontend)
- [Docker](https://docs.docker.com/get-docker/) (if using containerized setup)

---

### Clone the Repo
```bash
git clone https://github.com/UdayKiranKalli/Phonalynx.git
cd Phonalynx
Backend + Frontend Setup
1. Install Python deps
bash
Copy
Edit
pip install -r requirements.txt
2. (Optional) Rebuild Frontend
If you want to edit the React app, go into your React source folder, run:

bash
Copy
Edit
npm install
npm run build
The build will output to /dist/, which Flask will serve automatically.

3. Run Flask
bash
Copy
Edit
flask run
Run with Docker
bash
Copy
Edit
docker-compose up --build
📁 Project Structure
bash
Copy
Edit
Phonalynx/
│── app.py              # Flask backend entry
│── dist/               # React production build served by Flask
│── requirements.txt
│── docker-compose.yml
│── README.md
🔑 Environment Variables
Create a .env file in the root with:

env
Copy
Edit
SECRET_KEY=your_secret_key
DATABASE_URL=postgresql://user:password@localhost:5432/phonalynx
GITHUB_CLIENT_ID=xxxx
GITHUB_CLIENT_SECRET=xxxx
🤝 Contributing
Contributions, issues, and feature requests are welcome!
Feel free to fork and submit PRs.

📬 Contact
👤 Uday Kiran

GitHub: @UdayKiranKalli

LinkedIn:https://www.linkedin.com/in/kalliudaykiran

Email: kalliudaykiran@gmail.com

⭐ If you like this project, don’t forget to star the repo!

pgsql
Copy
Edit

---

✅ This version is **aligned with your structure** (`dist/` being served by Flask).  

