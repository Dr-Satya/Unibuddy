# 🚀 UniBuddy - Quick Start Guide

## What You Have Now

A complete MERN authentication system with:
- ✅ Professional dark-themed UI
- ✅ Modal-based authentication (Login, Signup, Forgot Password)
- ✅ Full CRUD Admin Panel for student management
- ✅ Role-Based Access Control (RBAC)
- ✅ Email verification
- ✅ Password reset
- ✅ JWT authentication with HTTP-only cookies

---

## 🏃 Quick Start (3 Steps)

### Step 1: Setup Environment
Create `.env` file in root directory:
```env
PORT=5000
CLIENT_URL=http://localhost:5173
MONGO_URI=your_mongodb_connection_string
JWT_SECRET=your_super_secret_jwt_key
MAILTRAP_TOKEN=your_mailtrap_token
MAILTRAP_ENDPOINT=https://send.api.mailtrap.io/
EMAIL_FROM=noreply@unibuddy.com
EMAIL_FROM_NAME=UniBuddy
```

### Step 2: Install & Run
```bash
# Install dependencies
npm install
cd frontend && npm install && cd ..

# Start backend (Terminal 1)
npm run dev

# Start frontend (Terminal 2)
cd frontend
npm run dev
```

### Step 3: Access Application
- Frontend: http://localhost:5173
- Backend: http://localhost:5000
- Admin Panel: http://localhost:5173/admin

---

## 👤 Create Admin User

### Method 1: Using MongoDB Compass/Shell
```javascript
// Connect to your database
use your_database_name

// Update user to admin
db.users.updateOne(
  { email: "youremail@gdgu.org" },
  { $set: { role: "ADMIN" } }
)
```

### Method 2: Update Admin Whitelist
Edit `backend/config/adminWhitelist.js`:
```javascript
export const adminEmails = [
  'admin@gdgu.org',
  'youremail@gdgu.org'  // Add your email
];
```

---

## 🎯 Key Features to Test

### 1. Authentication Flow
1. Go to http://localhost:5173
2. Click "Sign Up" button (modal opens)
3. Fill form and submit
4. Check email for verification code
5. Enter code on verification page
6. Redirected to dashboard

### 2. Modal Authentication
```javascript
// Use AuthButton component anywhere
import AuthButton from './components/AuthButton';

<AuthButton type="login" />
<AuthButton type="signup" />
<AuthButton type="forgot-password" />
```

### 3. Admin Panel
1. Login with admin account
2. Navigate to `/admin`
3. View student statistics
4. Add new student
5. Search/filter students
6. Edit/delete students
7. Export to CSV

---

## 📁 Project Structure

```
Unibuddy-MERN-Authentication/
├── backend/
│   ├── controllers/      # Business logic
│   ├── models/          # MongoDB schemas
│   ├── routes/          # API endpoints
│   ├── middleware/      # Auth middleware
│   └── mailtrap/        # Email service
├── frontend/
│   ├── src/
│   │   ├── components/  # Reusable components
│   │   ├── pages/       # Page components
│   │   ├── store/       # Zustand state
│   │   └── context/     # React context
│   └── public/
└── Documentation/
    ├── COMPLETE-SYSTEM-GUIDE.md
    ├── RBAC-DOCUMENTATION.md
    ├── ADMIN-PANEL-GUIDE.md
    ├── INTERVIEW-PREP.md
    └── FIXES-APPLIED.md
```

---

## 🔑 Important URLs

### Frontend Routes
- `/` - Landing page
- `/login` - Login page (direct access)
- `/signup` - Signup page (direct access)
- `/verify-email` - Email verification
- `/forgot-password` - Forgot password
- `/reset-password/:token` - Reset password
- `/dashboard` - User dashboard (protected)
- `/admin` - Admin panel (admin only)
- `/update-profile` - Update profile (protected)
- `/demo` - Demo page with auth buttons

### Backend API Endpoints
- `POST /api/auth/signup` - Register user
- `POST /api/auth/login` - Login user
- `POST /api/auth/logout` - Logout user
- `POST /api/auth/verify-email` - Verify email
- `POST /api/auth/forgot-password` - Request reset
- `POST /api/auth/reset-password/:token` - Reset password
- `GET /api/auth/check-auth` - Check auth status
- `GET /api/students` - Get all students (admin)
- `POST /api/students` - Add student (admin)
- `PUT /api/students/:id` - Update student (admin)
- `DELETE /api/students/:id` - Delete student (admin)

---

## 🎨 UI Theme Colors

```css
/* Background */
bg-slate-950, bg-slate-900, bg-slate-800

/* Accents */
from-blue-600 to-purple-600
from-blue-400 via-purple-400 to-indigo-400

/* Text */
text-white, text-slate-300, text-slate-400

/* Borders */
border-slate-800, border-slate-700

/* Status Colors */
green-500 (Active)
yellow-500 (Inactive)
red-500 (Suspended)
```

---

## 🔒 Security Features

1. **Password Hashing**: bcrypt with 10 salt rounds
2. **JWT Tokens**: HTTP-only cookies, 7-day expiry
3. **Email Validation**: Only @gdgu.org domain
4. **CSRF Protection**: SameSite cookie attribute
5. **XSS Protection**: Input sanitization
6. **RBAC**: Role-based access control
7. **Token Verification**: Middleware on protected routes

---

## 🐛 Common Issues & Solutions

### Issue: 500 Error When Adding Student
**Solution**: Restart backend server to load updated routes
```bash
# Stop server (Ctrl+C)
npm run dev
```

### Issue: Modal Not Opening
**Solution**: Check if AuthModalContext is wrapped around App
```javascript
// main.jsx
<AuthModalProvider>
  <App />
</AuthModalProvider>
```

### Issue: Can't Access Admin Panel
**Solution**: Update user role to ADMIN in database
```javascript
db.users.updateOne(
  { email: "your@email.com" },
  { $set: { role: "ADMIN" } }
)
```

### Issue: Email Not Sending
**Solution**: Verify Mailtrap credentials in .env
- Check MAILTRAP_TOKEN
- Verify endpoint URL
- Test Mailtrap account

---

## 📚 Documentation Files

1. **COMPLETE-SYSTEM-GUIDE.md** - Full system documentation
2. **RBAC-DOCUMENTATION.md** - Role-based access control guide
3. **ADMIN-PANEL-GUIDE.md** - Admin panel features
4. **INTERVIEW-PREP.md** - Interview questions & answers
5. **FIXES-APPLIED.md** - Recent bug fixes
6. **QUICK-START.md** - This file

---

## 🎓 Learning Resources

### Technologies Used
- **React**: https://react.dev
- **Vite**: https://vitejs.dev
- **Tailwind CSS**: https://tailwindcss.com
- **Framer Motion**: https://www.framer.com/motion
- **Zustand**: https://zustand-demo.pmnd.rs
- **Express**: https://expressjs.com
- **MongoDB**: https://www.mongodb.com
- **Mongoose**: https://mongoosejs.com
- **JWT**: https://jwt.io

### Key Concepts
- JWT Authentication
- HTTP-only Cookies
- Role-Based Access Control (RBAC)
- Password Hashing (bcrypt)
- Email Verification
- Password Reset Flow
- Modal-based UI
- State Management (Zustand)
- React Context API
- Protected Routes
- Middleware
- MongoDB Schemas
- RESTful API Design

---

## 🚀 Next Steps

1. **Test Everything**: Go through all features
2. **Customize**: Update colors, logos, text
3. **Add Features**: Implement additional functionality
4. **Deploy**: Deploy to production (Vercel, Heroku, etc.)
5. **Monitor**: Set up error tracking and analytics

---

## 💡 Pro Tips

1. **Use AuthButton**: Instead of creating forms, use `<AuthButton type="login" />` for instant modal popups
2. **Check Console**: Always check browser console and backend terminal for errors
3. **Test with Real Email**: Use real email for testing verification flow
4. **Admin First**: Create admin user before testing admin panel
5. **Read Docs**: Check COMPLETE-SYSTEM-GUIDE.md for detailed info

---

## 🎉 You're Ready!

Your authentication system is production-ready with:
- ✅ Secure authentication
- ✅ Professional UI
- ✅ Admin panel
- ✅ Email verification
- ✅ Password reset
- ✅ RBAC
- ✅ Full documentation

**Start building amazing features on top of this foundation!**

---

**Need Help?** Check the documentation files or review the code comments.

**Happy Coding! 🚀**
