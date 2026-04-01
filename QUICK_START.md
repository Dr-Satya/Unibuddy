# 🚀 UniBuddy - Quick Start Guide

## ⚡ Fastest Way to Get Started

### Step 1: Start All Services

Choose one of these methods:

#### Option 1: Using Batch File (Windows - Easiest)
```cmd
start-all.bat
```

#### Option 2: Using Python Script (Cross-platform)
```bash
python start_all.py
```

#### Option 3: Using PowerShell
```powershell
.\start-all.ps1
```

#### Option 4: Manual Start (Most Reliable)

**Terminal 1 - Auth Backend:**
```bash
cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
npm start
```

**Terminal 2 - Chatbot Backend:**
```bash
cd backend
python api.py
```

**Terminal 3 - Frontend:**
```bash
npm run dev
```

This opens 3 windows/terminals:
- 🟢 Auth Backend (Port 5000)
- 🟢 Chatbot Backend (Port 9000)
- 🟢 Frontend (Port 5173)

### Step 2: Open Browser
Go to: **http://localhost:5173**

### Step 3: Test the Flow

#### As a Student:
1. Click **"Sign Up"**
2. Fill the form:
   - Email: `student@gdgu.org`
   - Password: `Test@1234`
   - Fill other required fields
3. Click **"Sign Up"**
4. Enter verification code from email
5. ✅ You're now on the chatbot page!
6. Click the chatbot icon (bottom-right)
7. Start chatting!

#### As an Admin:
1. Click **"Sign Up"**
2. Fill the form:
   - Email: `saafin@gdgu.org` (or any admin email)
   - Password: `Test@1234`
   - Fill other required fields
3. Click **"Sign Up"**
4. Enter verification code from email
5. ✅ You're now on the admin panel!

## 🎯 What Happens?

### Before Login:
```
Landing Page → Click "Get Started" → Login/Signup
```

### After Login (Student):
```
Login → Verify Email → Chatbot Page ✅
```

### After Login (Admin):
```
Login → Verify Email → Admin Panel ✅
```

## 🔐 Admin Emails

These emails get admin access:
- saafin@gdgu.org
- 230160223057.saafin@gdgu.org
- samkit@gdgu.org
- kiyosha@gdgu.org

All other @gdgu.org emails get student access.

## 📍 Access Points

- **Frontend**: http://localhost:5173
- **Auth API**: http://localhost:5000
- **Chatbot API**: http://localhost:9000

## 🛑 Stop All Services

Close all 3 PowerShell windows that opened.

## ❓ Problems?

### Auth backend won't start?
- Check if MongoDB is running
- Check `.env` file in `backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/`

### Chatbot backend won't start?
- Check if Python is installed
- Run: `pip install -r requirements.txt` in `backend/` folder

### Frontend won't start?
- Run: `npm install` in `UniBuddy/` folder

## 📚 More Info

- Full guide: `README_INTEGRATION.md`
- Technical details: `INTEGRATION_GUIDE.md`
- Completion status: `INTEGRATION_COMPLETE.md`

---

**That's it! You're ready to go! 🎉**
