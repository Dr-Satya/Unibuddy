# 🔗 UniBuddy Platform - Authentication & Chatbot Integration

## Overview
UniBuddy integrates a **MERN authentication system** with a **Python RAG chatbot** to create a secure, role-based university assistant platform.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UniBuddy Platform                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Frontend (React + TypeScript) - Port 5173              │
│  ├── Auth Pages (Login, Signup, Verification)           │
│  ├── HomePage (Students)                                │
│  ├── Admin Panel (Admins)                               │
│  └── Chatbot Widget (All authenticated users)           │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Auth Backend (Node.js + Express) - Port 5000           │
│  ├── User Authentication (JWT)                          │
│  ├── Email Verification (OTP)                           │
│  ├── Admin Whitelist                                    │
│  └── Student Management (CRUD)                          │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Chatbot Backend (Python + FastAPI) - Port 9000         │
│  ├── RAG System (Vector DB)                             │
│  ├── Faculty Information                                │
│  └── University Data Queries                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔐 Authentication Integration

### 1. **Backend Integration**
- **Used:** Complete MERN authentication backend (Port 5000)
- **Features:**
  - JWT token-based authentication
  - Email verification with OTP
  - Password reset flow
  - Admin whitelist system
  - Student CRUD operations

### 2. **Frontend Integration**
- **Copied:** Auth components from MERN frontend to UniBuddy/src
- **Adapted:** TypeScript conversion + modal support
- **Components:**
  - LoginPage, SignupPage, ForgotPasswordPage
  - EmailVerificationPage, ResetPasswordPage
  - AdminPanel, AuthStore, ProtectedRoute

### 3. **Key Integration Points**

```typescript
// AuthStore connects to Auth Backend
const API_URL = 'http://localhost:5000/api/auth';

// Chatbot connects to Python Backend
const API_URL = 'http://127.0.0.1:9000/chat';
```

---

## 🤖 Chatbot Integration

### Access Control
```
Not Logged In → Chatbot icon shows 🔒 → Click → Login Modal
Logged In (Student) → Chatbot accessible on HomePage
Logged In (Admin) → Chatbot accessible on Admin Panel
```

### Implementation
```typescript
// Chatbot.tsx
const handleChatbotClick = () => {
  if (!isAuthenticated) {
    openModal('login'); // Show login modal
  } else {
    setOpen(true); // Open chatbot
  }
};
```

---

## 👥 User Flows

### Student Flow
```
1. Click Signup → Modal opens
2. Fill form → Submit
3. Email Verification → Enter OTP
4. Verified → Redirect to HomePage
5. Chatbot accessible (no re-login)
6. Auto-created in Student database (visible in Admin Panel)
```

### Admin Flow
```
1. Login with whitelisted email
2. Email Verification (if needed)
3. Redirect to Admin Panel
4. Manage students + Use chatbot
```

---

## 🔄 Data Flow

### Signup Process
```javascript
User Signup
    ↓
1. Create User record (authentication)
    ↓
2. Create Student record (if role = STUDENT)
    ↓
3. Send verification email
    ↓
4. User verifies → Access granted
```

### Chatbot Query
```javascript
User sends message
    ↓
Frontend → POST /chat (Port 9000)
    ↓
Python RAG system processes
    ↓
Response → Display in chatbot
```

---

## 🎯 Key Features

### Authentication
- ✅ JWT tokens (httpOnly cookies)
- ✅ Email verification (OTP)
- ✅ Password reset
- ✅ Admin whitelist
- ✅ Role-based access

### Chatbot
- ✅ RAG-based responses
- ✅ Faculty information
- ✅ University data queries
- ✅ Session management
- ✅ Protected access

### Admin Panel
- ✅ View all students
- ✅ Add/Edit/Delete students
- ✅ Export to CSV
- ✅ Search & filter
- ✅ Status management

---

## 🚀 Running the Platform

```bash
# Start all services
python start_all.py

# Services started:
# - Auth Backend: http://localhost:5000
# - Chatbot Backend: http://localhost:9000
# - Frontend: http://localhost:5173
```

---

## 📊 Integration Summary

| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| Frontend | React + TS | 5173 | UI & User interaction |
| Auth Backend | Node.js | 5000 | Authentication & User management |
| Chatbot Backend | Python | 9000 | RAG chatbot responses |

### Communication
- Frontend ↔ Auth Backend: REST API (JWT cookies)
- Frontend ↔ Chatbot Backend: REST API (session-based)
- Auth Backend ↔ Database: MongoDB
- Chatbot Backend ↔ Vector DB: FAISS

---

## 🔒 Security

- **Authentication:** JWT tokens in httpOnly cookies
- **Authorization:** Role-based access control (RBAC)
- **Validation:** Email domain restriction (@gdgu.org)
- **Password:** Bcrypt hashing (10 rounds)
- **Admin Access:** Server-side whitelist only

---

## ✨ Unique Features

1. **Modal-based Auth:** Login/Signup in modals (no page reload)
2. **Auto Student Creation:** Students auto-added to Admin Panel
3. **Dual Model System:** User (auth) + Student (data)
4. **Integrated Chatbot:** Seamless chatbot access after login
5. **Role-based UI:** Different interfaces for students vs admins

---

## 📝 Quick Explanation Points

**Q: How is authentication integrated?**
> We use a complete MERN authentication backend (Port 5000) with JWT tokens. The frontend components were adapted from the auth system and integrated into the main UniBuddy React app with TypeScript and modal support.

**Q: How does the chatbot work with auth?**
> The chatbot checks authentication status. If not logged in, it shows a login modal. Once authenticated, users can access the chatbot which connects to a separate Python backend (Port 9000) for RAG-based responses.

**Q: What about admin access?**
> Admins are managed through a server-side whitelist. When a whitelisted email signs up, they automatically get admin role and access to the Admin Panel where they can manage all students.

**Q: How are students managed?**
> When a student signs up, two records are created: one in the User collection (for authentication) and one in the Student collection (for academic data). This allows admins to manage student information while keeping auth separate.

---

**Built with ❤️ for GD Goenka University**
