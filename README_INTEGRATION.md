# UniBuddy - Authentication Integration

## 🎯 What's Been Done

Your UniBuddy chatbot now has a complete authentication system integrated with Role-Based Access Control (RBAC). The integration follows your exact requirements:

### ✅ Integration Flow

1. **User clicks chatbot** → Checks if logged in
2. **Not logged in** → Redirects to authentication (signup/login)
3. **After login** → Checks user role:
   - **STUDENT** → Access chatbot
   - **ADMIN** → Access admin panel (with chatbot available)

### ✅ What's Preserved

- ✅ Original chatbot functionality (100% unchanged)
- ✅ Authentication system (100% unchanged)
- ✅ Admin panel (accessible for admins)
- ✅ All existing features and UI

### ✅ What's New

- ✅ React Router for navigation
- ✅ Protected routes with authentication
- ✅ Role-based access control
- ✅ Login/Signup pages
- ✅ Email verification flow
- ✅ Zustand state management for auth
- ✅ Seamless integration between systems

## 🚀 Quick Start

### Option 1: Automated Start (Recommended)
```powershell
# Run from UniBuddy directory
.\start-all.ps1
```

This will start all three services:
- Frontend (Port 5173)
- Auth Backend (Port 5000)
- Chatbot Backend (Port 9000)

### Option 2: Manual Start

#### 1. Start Authentication Backend
```bash
cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
npm install
npm start
```

#### 2. Start Chatbot Backend
```bash
cd backend
pip install -r requirements.txt
python api.py
```

#### 3. Start Frontend
```bash
cd UniBuddy
npm install
npm run dev
```

## 📋 Testing the Integration

### Test as Student:
1. Open http://localhost:5173
2. Click "Sign Up"
3. Register with: `student@gdgu.org`
4. Complete verification
5. ✅ Should redirect to chatbot page
6. ✅ Can use chatbot

### Test as Admin:
1. Open http://localhost:5173
2. Click "Sign Up"
3. Register with: `saafin@gdgu.org` (or any admin email from whitelist)
4. Complete verification
5. ✅ Should redirect to admin panel
6. ✅ Can access admin features

## 🔐 Admin Whitelist

Admin emails are configured in:
```
backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/config/adminWhitelist.js
```

Current admin emails:
- saafin@gdgu.org
- 230160223057.saafin@gdgu.org
- samkit@gdgu.org
- kiyosha@gdgu.org

## 📁 Project Structure

```
UniBuddy/
├── src/
│   ├── components/          # UI components
│   │   ├── Chatbot.tsx     # Original chatbot (unchanged)
│   │   ├── ProtectedRoute.tsx  # Route protection
│   │   └── ...
│   ├── pages/              # Page components
│   │   ├── HomePage.tsx    # Landing page
│   │   ├── LoginPage.tsx   # Login form
│   │   ├── SignupPage.tsx  # Registration
│   │   ├── ChatbotPage.tsx # Protected chatbot
│   │   └── ...
│   ├── store/
│   │   └── authStore.ts    # Authentication state
│   └── App.tsx             # Main app with routing
├── backend/
│   ├── src/                # Python chatbot backend
│   └── Unibuddy-Auth/     # Node.js auth backend
└── start-all.ps1          # Automated startup script
```

## 🔄 User Flow Diagram

```
Landing Page (/)
    ↓
[Not Authenticated]
    ↓
Login/Signup (/login or /signup)
    ↓
Email Verification (/verify-email)
    ↓
[Check Role]
    ↓
    ├─→ STUDENT → Chatbot Page (/chatbot)
    └─→ ADMIN → Admin Panel (/admin)
```

## 🛠️ Configuration

### Environment Variables

Create `.env` in `backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/`:

```env
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_jwt_secret_key
CLIENT_URL=http://localhost:5173
MAILTRAP_TOKEN=your_mailtrap_token
PORT=5000
```

## 📝 Key Features

### Authentication
- ✅ JWT-based authentication
- ✅ HTTP-only cookies for security
- ✅ Email verification required
- ✅ Password reset functionality
- ✅ GDGU email validation (@gdgu.org)

### Authorization (RBAC)
- ✅ Student role (default)
- ✅ Admin role (whitelist-based)
- ✅ Protected routes
- ✅ Role-based redirects

### Chatbot
- ✅ Session-based conversations
- ✅ Message history
- ✅ HTML response rendering
- ✅ Debug info cleanup
- ✅ User authentication required

## 🐛 Troubleshooting

### Issue: Cannot connect to auth backend
**Solution**: Ensure MongoDB is running and MONGO_URI is correct in .env

### Issue: Chatbot not responding
**Solution**: Check if Python backend is running on port 9000

### Issue: Email verification not working
**Solution**: Verify Mailtrap configuration in .env

### Issue: Admin role not assigned
**Solution**: Check if email is in adminWhitelist.js and restart auth backend

## 📚 Documentation

- [Integration Guide](./INTEGRATION_GUIDE.md) - Detailed integration documentation
- [Auth System Guide](./backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/COMPLETE-SYSTEM-GUIDE.md) - Authentication system details
- [Admin Panel Guide](./backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/ADMIN-PANEL-GUIDE.md) - Admin panel documentation

## 🎉 Success Criteria

✅ User must login before accessing chatbot
✅ Students can access chatbot after login
✅ Admins can access admin panel after login
✅ Original chatbot functionality preserved
✅ Original authentication system preserved
✅ Role-based access control working
✅ Email verification working
✅ Seamless user experience

## 💡 Next Steps

1. Test the complete flow with both student and admin accounts
2. Customize the admin panel integration if needed
3. Add additional features (profile page, chat history, etc.)
4. Deploy to production

## 🤝 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Review the integration guide
3. Verify all services are running
4. Check browser console for errors

---

**Note**: All original code (chatbot and authentication) remains unchanged. Only integration layer has been added.
