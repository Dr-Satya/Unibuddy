# 🎯 START HERE - UniBuddy Integration

## Welcome! 👋

Your UniBuddy chatbot now has authentication integrated with Role-Based Access Control (RBAC).

## 🚀 Quick Start (Choose One Method)

### Method 1: Batch File (Windows - Easiest) ⭐
```cmd
start-all.bat
```
Double-click the file or run from command prompt.

### Method 2: Python Script (Cross-platform)
```bash
python start_all.py
```

### Method 3: Manual (Most Reliable)
See `MANUAL_START.md` for step-by-step instructions.

Open 3 terminals and run:
1. `cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend && npm start`
2. `cd backend && python api.py`
3. `npm run dev`

## 📍 Access the App

Once all services are running, open:
```
http://localhost:5173
```

## 🧪 Test It

### As Student:
1. Click "Sign Up"
2. Email: `student@gdgu.org`
3. Password: `Test@1234`
4. Fill other fields
5. Verify email
6. ✅ Chatbot access!

### As Admin:
1. Click "Sign Up"
2. Email: `saafin@gdgu.org`
3. Password: `Test@1234`
4. Fill other fields
5. Verify email
6. ✅ Admin panel access!

## 📚 Documentation

- **MANUAL_START.md** - Step-by-step manual start guide
- **QUICK_START.md** - Quick start with all options
- **README_INTEGRATION.md** - Complete setup guide
- **INTEGRATION_COMPLETE.md** - What was done
- **INTEGRATION_GUIDE.md** - Technical details
- **ARCHITECTURE.md** - System architecture

## ❓ Problems?

### Services won't start?
1. Check if MongoDB is running
2. Check if ports 5000, 9000, 5173 are free
3. See `MANUAL_START.md` for troubleshooting

### Need help?
Check the troubleshooting section in `MANUAL_START.md`

## ✅ What's Working

- ✅ Authentication with JWT
- ✅ Email verification
- ✅ Role-based access (Student/Admin)
- ✅ Protected chatbot
- ✅ Admin panel access
- ✅ Original chatbot preserved
- ✅ Original auth system preserved

## 🎉 You're Ready!

Everything is set up and ready to go. Just start the services and test it out!

---

**Quick Links:**
- Frontend: http://localhost:5173
- Auth API: http://localhost:5000
- Chatbot API: http://localhost:9000
