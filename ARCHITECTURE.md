# 🏗️ UniBuddy Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     UniBuddy System                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Frontend   │  │  Auth Backend│  │Chatbot Backend│   │
│  │  (React +    │  │  (Node.js +  │  │  (Python +   │    │
│  │   Vite)      │  │   Express)   │  │   FastAPI)   │    │
│  │              │  │              │  │              │    │
│  │ Port: 5173   │  │ Port: 5000   │  │ Port: 9000   │    │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │  MongoDB    │                        │
│                    │  Database   │                        │
│                    └─────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### Frontend (React + TypeScript)

```
src/
├── components/
│   ├── Chatbot.tsx              [Original chatbot widget]
│   ├── Navbar.tsx               [Navigation with auth buttons]
│   ├── Home.tsx                 [Landing page hero]
│   ├── Features.tsx             [Features section]
│   ├── Footer.tsx               [Footer section]
│   ├── ProtectedRoute.tsx       [Route protection HOC]
│   ├── RedirectAuthenticatedUser.tsx [Auth redirect logic]
│   └── LoadingSpinner.tsx       [Loading state]
│
├── pages/
│   ├── HomePage.tsx             [Landing page wrapper]
│   ├── LoginPage.tsx            [Login form]
│   ├── SignupPage.tsx           [Registration form]
│   ├── EmailVerificationPage.tsx [Email verification]
│   └── ChatbotPage.tsx          [Protected chatbot page]
│
├── store/
│   └── authStore.ts             [Zustand auth state]
│
├── App.tsx                      [Main app with routing]
└── main.tsx                     [React entry point]
```

### Backend Services

#### Auth Backend (Node.js + Express)
```
backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend/
├── controllers/
│   ├── auth.controller.js       [Auth logic]
│   └── user.controller.js       [User management]
├── models/
│   ├── user.model.js            [User schema]
│   └── student.model.js         [Student schema]
├── routes/
│   ├── auth.route.js            [Auth routes]
│   ├── user.routes.js           [User routes]
│   └── student.route.js         [Student routes]
├── middleware/
│   └── verifyToken.js           [JWT verification]
└── config/
    └── adminWhitelist.js        [Admin emails]
```

#### Chatbot Backend (Python + FastAPI)
```
backend/
├── src/
│   ├── api_adapter.py           [Session management]
│   ├── main.py                  [RAG system]
│   ├── models.py                [Data models]
│   └── services.py              [Business logic]
└── api.py                       [FastAPI entry point]
```

## Data Flow

### Authentication Flow

```
┌──────────┐
│  User    │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Login/Signup   │
│     Form        │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Auth Store     │
│  (Zustand)      │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  POST /api/auth │
│  /login         │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Auth Backend   │
│  (Node.js)      │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  MongoDB        │
│  Verify User    │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  JWT Token      │
│  (HTTP Cookie)  │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Check Role     │
│  STUDENT/ADMIN  │
└────┬────────────┘
     │
     ├─→ STUDENT → /chatbot
     └─→ ADMIN → /admin
```

### Chatbot Interaction Flow

```
┌──────────┐
│  User    │
│ (Logged) │
└────┬─────┘
     │
     ▼
┌─────────────────┐
│  Chatbot Page   │
│  /chatbot       │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Chatbot Widget │
│  (Bottom-right) │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  User Message   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  POST /chat     │
│  {message,      │
│   session_id}   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Chatbot Backend│
│  (Python)       │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  RAG System     │
│  (Retrieval +   │
│   Generation)   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  AI Response    │
│  {reply, data}  │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Display in     │
│  Chat Widget    │
└─────────────────┘
```

## Route Protection

### Public Routes
```
/                    → HomePage (landing page)
/login               → LoginPage
/signup              → SignupPage
```

### Protected Routes
```
/chatbot             → ChatbotPage (requires auth)
/admin               → AdminPanel (requires auth + ADMIN role)
/verify-email        → EmailVerificationPage (requires auth)
```

### Route Guard Logic

```typescript
// ProtectedRoute Component
if (!isAuthenticated) {
  return <Navigate to="/login" />
}

if (requireAdmin && user.role !== 'ADMIN') {
  return <Navigate to="/chatbot" />
}

return <>{children}</>
```

## State Management

### Auth Store (Zustand)

```typescript
interface AuthState {
  user: User | null
  isAuthenticated: boolean
  error: string | null
  isLoading: boolean
  isCheckingAuth: boolean
  
  // Actions
  signup()
  login()
  logout()
  verifyEmail()
  checkAuth()
  forgotPassword()
  resetPassword()
}
```

### User Object

```typescript
interface User {
  _id: string
  email: string
  role: 'STUDENT' | 'ADMIN'
  isVerified: boolean
  fatherName?: string
  motherName?: string
  contactNumber?: string
}
```

## Security Architecture

### Authentication
```
┌─────────────────┐
│  User Login     │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Password Hash  │
│  (bcrypt)       │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  JWT Token      │
│  Generation     │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  HTTP-Only      │
│  Cookie         │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Secure         │
│  Storage        │
└─────────────────┘
```

### Authorization (RBAC)
```
┌─────────────────┐
│  User Request   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Extract JWT    │
│  from Cookie    │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Verify Token   │
│  (JWT Secret)   │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│  Check Role     │
│  (User.role)    │
└────┬────────────┘
     │
     ├─→ STUDENT → Allow chatbot
     ├─→ ADMIN → Allow admin panel
     └─→ Invalid → Deny access
```

## Database Schema

### User Collection (MongoDB)
```javascript
{
  _id: ObjectId,
  email: String (unique, @gdgu.org),
  password: String (hashed),
  role: String (STUDENT | ADMIN),
  isVerified: Boolean,
  fatherName: String,
  motherName: String,
  contactNumber: String,
  photo: String (base64),
  collegeIdCard: String (base64),
  verificationToken: String,
  verificationTokenExpiresAt: Date,
  resetPasswordToken: String,
  resetPasswordExpiresAt: Date,
  lastLogin: Date,
  createdAt: Date,
  updatedAt: Date
}
```

### Student Collection (MongoDB)
```javascript
{
  _id: ObjectId,
  name: String,
  email: String (unique, @gdgu.org),
  rollNo: String (unique),
  department: String,
  year: String (1-4),
  fatherName: String,
  motherName: String,
  contactNumber: String,
  address: String,
  dateOfBirth: Date,
  gender: String,
  photo: String,
  collegeIdCard: String,
  status: String (ACTIVE | INACTIVE | SUSPENDED),
  createdAt: Date,
  updatedAt: Date
}
```

## API Endpoints

### Auth API (Port 5000)
```
POST   /api/auth/signup              Register new user
POST   /api/auth/login               Login user
POST   /api/auth/logout              Logout user
POST   /api/auth/verify-email        Verify email with code
POST   /api/auth/forgot-password     Request password reset
POST   /api/auth/reset-password/:token  Reset password
GET    /api/auth/check-auth          Check auth status
```

### Chatbot API (Port 9000)
```
POST   /chat                         Send message to chatbot
       Body: { message, session_id }
       Response: { reply, data }
```

### Admin API (Port 5000)
```
GET    /api/students                 Get all students (admin)
POST   /api/students                 Add student (admin)
PUT    /api/students/:id             Update student (admin)
DELETE /api/students/:id             Delete student (admin)
```

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Production Setup                     │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Frontend   │  │  Auth Backend│  │Chatbot Backend│ │
│  │   (Vercel/   │  │  (Heroku/    │  │  (Railway/   │ │
│  │   Netlify)   │  │   Railway)   │  │   Render)    │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                 │                  │          │
│         └─────────────────┴──────────────────┘          │
│                           │                             │
│                    ┌──────▼──────┐                     │
│                    │  MongoDB    │                     │
│                    │   Atlas     │                     │
│                    └─────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

## Technology Stack

### Frontend
- React 19.2.0
- TypeScript 5.9.3
- Vite 7.2.4
- React Router DOM 6.30.2
- Zustand 4.5.7
- Axios 1.13.2
- React Hot Toast 2.6.0
- Tailwind CSS 3.4.19

### Auth Backend
- Node.js
- Express.js
- MongoDB + Mongoose
- JWT (jsonwebtoken)
- bcryptjs
- Mailtrap (email service)

### Chatbot Backend
- Python 3.x
- FastAPI
- SQLAlchemy
- Sentence Transformers
- FAISS
- Groq API
- LangChain

## Performance Considerations

### Frontend
- Code splitting with React Router
- Lazy loading of components
- Optimized bundle size
- Efficient state management with Zustand

### Backend
- JWT token caching
- Database connection pooling
- API response caching
- Efficient query optimization

### Chatbot
- Session-based conversations
- Vector database for fast retrieval
- Streaming responses
- Efficient RAG pipeline

---

**This architecture ensures scalability, security, and maintainability! 🚀**
