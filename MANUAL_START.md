# 📖 Manual Start Guide - Step by Step

## Prerequisites

Before starting, make sure you have:
- ✅ Node.js installed (v16 or higher)
- ✅ Python installed (v3.8 or higher)
- ✅ MongoDB running (locally or Atlas)

## Step-by-Step Instructions

### Step 1: Open 3 Terminal Windows

You'll need 3 separate terminal/command prompt windows.

---

### Terminal 1: Start Auth Backend

1. Open a new terminal/command prompt
2. Navigate to the auth backend directory:
   ```bash
   cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
   ```

3. Install dependencies (first time only):
   ```bash
   npm install
   ```

4. Make sure you have a `.env` file with:
   ```env
   MONGO_URI=your_mongodb_connection_string
   JWT_SECRET=your_jwt_secret
   CLIENT_URL=http://localhost:5173
   MAILTRAP_TOKEN=your_mailtrap_token
   PORT=5000
   ```

5. Start the server:
   ```bash
   npm start
   ```

6. ✅ You should see: `Server is running on port 5000`

---

### Terminal 2: Start Chatbot Backend

1. Open a new terminal/command prompt
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```

3. Install dependencies (first time only):
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   python api.py
   ```

5. ✅ You should see: `Uvicorn running on http://127.0.0.1:9000`

---

### Terminal 3: Start Frontend

1. Open a new terminal/command prompt
2. Make sure you're in the UniBuddy root directory

3. Install dependencies (first time only):
   ```bash
   npm install
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. ✅ You should see: `Local: http://localhost:5173/`

---

## Step 2: Open Browser

Open your browser and go to:
```
http://localhost:5173
```

## Step 3: Test the Application

### Create a Student Account:
1. Click "Sign Up"
2. Fill the form:
   - Email: `student@gdgu.org`
   - Password: `Test@1234`
   - Father's Name: `John Doe`
   - Mother's Name: `Jane Doe`
   - Contact: `9876543210`
   - Upload photo and ID card
3. Click "Sign Up"
4. Check your email for verification code
5. Enter the 6-digit code
6. ✅ You should be redirected to the chatbot page!

### Create an Admin Account:
1. Click "Sign Up"
2. Fill the form with:
   - Email: `saafin@gdgu.org` (or any admin email from whitelist)
   - Other required fields
3. Complete verification
4. ✅ You should be redirected to the admin panel!

## Troubleshooting

### Auth Backend Won't Start

**Error: Cannot connect to MongoDB**
- Solution: Make sure MongoDB is running
- Check your MONGO_URI in .env file
- If using MongoDB Atlas, check your IP whitelist

**Error: Port 5000 already in use**
- Solution: Kill the process using port 5000
- Windows: `netstat -ano | findstr :5000` then `taskkill /PID <PID> /F`
- Mac/Linux: `lsof -ti:5000 | xargs kill -9`

### Chatbot Backend Won't Start

**Error: Module not found**
- Solution: Install Python dependencies
- Run: `pip install -r requirements.txt`

**Error: Port 9000 already in use**
- Solution: Kill the process using port 9000
- Windows: `netstat -ano | findstr :9000` then `taskkill /PID <PID> /F`
- Mac/Linux: `lsof -ti:9000 | xargs kill -9`

### Frontend Won't Start

**Error: Cannot find module**
- Solution: Install npm dependencies
- Run: `npm install`

**Error: Port 5173 already in use**
- Solution: Kill the process or use a different port
- The Vite dev server will automatically suggest an alternative port

### Email Verification Not Working

**No email received**
- Solution: Check Mailtrap configuration
- Verify MAILTRAP_TOKEN in .env
- Check Mailtrap inbox at https://mailtrap.io

### Admin Role Not Assigned

**Logged in as student instead of admin**
- Solution: Check if your email is in the admin whitelist
- File: `backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/config/adminWhitelist.js`
- Add your email to the ADMIN_WHITELIST array
- Restart the auth backend

## Stopping the Services

To stop all services:
1. Go to each terminal window
2. Press `Ctrl + C`
3. Close the terminal windows

## Quick Reference

### Service URLs
- Frontend: http://localhost:5173
- Auth Backend: http://localhost:5000
- Chatbot Backend: http://localhost:9000

### Admin Emails (from whitelist)
- saafin@gdgu.org
- 230160223057.saafin@gdgu.org
- samkit@gdgu.org
- kiyosha@gdgu.org

### Test Credentials
You can create test accounts with any @gdgu.org email:
- Student: `student@gdgu.org`
- Admin: Use one of the whitelist emails above

## Next Steps

Once everything is running:
1. ✅ Test student signup and login
2. ✅ Test admin signup and login
3. ✅ Test chatbot functionality
4. ✅ Test admin panel features
5. ✅ Customize as needed

---

**Need more help? Check the other documentation files:**
- `QUICK_START.md` - Automated startup options
- `README_INTEGRATION.md` - Complete integration guide
- `INTEGRATION_GUIDE.md` - Technical details
- `ARCHITECTURE.md` - System architecture
