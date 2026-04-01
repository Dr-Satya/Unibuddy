# 🎓 UniBuddy - AI-Powered University Assistant

> Intelligent chatbot with authentication and role-based access control

## 🌟 Features

- 🤖 **AI Chatbot** - RAG-powered university information assistant
- 🔐 **Authentication** - Secure JWT-based authentication
- 👥 **RBAC** - Role-Based Access Control (Student/Admin)
- ✉️ **Email Verification** - Secure account verification
- 📊 **Admin Panel** - Complete student management system
- 🎨 **Modern UI** - Dark theme with smooth animations

## 🚀 Quick Start

### Prerequisites
- Node.js (v16+)
- Python (v3.8+)
- MongoDB (local or Atlas)

### Start the Application

**Option 1: Batch File (Windows)**
```cmd
start-all.bat
```

**Option 2: Python Script**
```bash
python start_all.py
```

**Option 3: Manual Start**
```bash
# Terminal 1 - Auth Backend
cd backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend
npm start

# Terminal 2 - Chatbot Backend
cd backend
python api.py

# Terminal 3 - Frontend
npm run dev
```

### Access the App
Open http://localhost:5173

## 📖 Documentation

| Document | Description |
|----------|-------------|
| **START_HERE.md** | 👈 Start here! Quick overview |
| **MANUAL_START.md** | Step-by-step manual start guide |
| **QUICK_START.md** | Quick start with all options |
| **README_INTEGRATION.md** | Complete integration guide |
| **INTEGRATION_COMPLETE.md** | What was implemented |
| **INTEGRATION_GUIDE.md** | Technical documentation |
| **ARCHITECTURE.md** | System architecture |

## 🎯 User Flow

```
Landing Page → Login/Signup → Email Verification
                                      ↓
                              Check User Role
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
              STUDENT Role                        ADMIN Role
                    ↓                                   ↓
            Chatbot Access                      Admin Panel
```

## 🧪 Test Accounts

### Student Account
- Email: `student@gdgu.org`
- Password: `Test@1234`
- Access: Chatbot only

### Admin Account
- Email: `saafin@gdgu.org` (or any admin from whitelist)
- Password: `Test@1234`
- Access: Admin panel + Chatbot

## 🔐 Admin Whitelist

Admin emails are configured in:
```
backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/config/adminWhitelist.js
```

Current admins:
- saafin@gdgu.org
- 230160223057.saafin@gdgu.org
- samkit@gdgu.org
- kiyosha@gdgu.org

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                  UniBuddy System                    │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Frontend │  │   Auth   │  │ Chatbot  │        │
│  │  React   │  │  Node.js │  │  Python  │        │
│  │  :5173   │  │  :5000   │  │  :9000   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       └─────────────┴──────────────┘               │
│                     │                              │
│              ┌──────▼──────┐                       │
│              │   MongoDB   │                       │
│              └─────────────┘                       │
└─────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

### Frontend
- React 19.2.0
- TypeScript 5.9.3
- Vite 7.2.4
- React Router DOM
- Zustand (State Management)
- Tailwind CSS

### Backend (Auth)
- Node.js + Express
- MongoDB + Mongoose
- JWT Authentication
- Mailtrap (Email)

### Backend (Chatbot)
- Python + FastAPI
- RAG (Retrieval-Augmented Generation)
- FAISS Vector Database
- Groq API

## 📦 Project Structure

```
UniBuddy/
├── src/                    # Frontend source
│   ├── components/         # React components
│   ├── pages/             # Page components
│   └── store/             # State management
├── backend/
│   ├── src/               # Chatbot backend
│   └── Unibuddy-Auth/    # Auth backend
├── start-all.bat          # Windows startup
├── start_all.py           # Python startup
└── start-all.ps1          # PowerShell startup
```

## 🔧 Environment Setup

Create `.env` in `backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/`:

```env
MONGO_URI=mongodb://localhost:27017/unibuddy
JWT_SECRET=your_secret_key_here
CLIENT_URL=http://localhost:5173
MAILTRAP_TOKEN=your_mailtrap_token
PORT=5000
```

## 🐛 Troubleshooting

### Services won't start?
1. Check MongoDB is running
2. Check ports 5000, 9000, 5173 are available
3. Install dependencies: `npm install` and `pip install -r requirements.txt`

### Email verification not working?
1. Check Mailtrap configuration
2. Verify MAILTRAP_TOKEN in .env

### Admin role not assigned?
1. Check email is in adminWhitelist.js
2. Restart auth backend

See `MANUAL_START.md` for detailed troubleshooting.

## ✅ Features Implemented

- ✅ JWT-based authentication
- ✅ Email verification with OTP
- ✅ Role-based access control
- ✅ Protected routes
- ✅ Student chatbot access
- ✅ Admin panel access
- ✅ Session management
- ✅ Password hashing
- ✅ CORS configuration
- ✅ Error handling
- ✅ Loading states
- ✅ Toast notifications

## 🎓 Usage

### For Students
1. Sign up with @gdgu.org email
2. Verify email
3. Access chatbot
4. Ask questions about university

### For Admins
1. Sign up with admin email
2. Verify email
3. Access admin panel
4. Manage students
5. Use chatbot

## 📝 API Endpoints

### Auth API (Port 5000)
- `POST /api/auth/signup` - Register
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout
- `POST /api/auth/verify-email` - Verify email
- `GET /api/auth/check-auth` - Check auth status

### Chatbot API (Port 9000)
- `POST /chat` - Send message

## 🚀 Deployment

Ready for production deployment:
- Frontend: Vercel, Netlify
- Auth Backend: Heroku, Railway
- Chatbot Backend: Railway, Render
- Database: MongoDB Atlas

## 📄 License

This project is part of GD Goenka University.

## 🤝 Contributing

This is an internal university project.

## 📞 Support

For issues or questions, check the documentation files or contact the development team.

---

**Made with ❤️ for GD Goenka University**
