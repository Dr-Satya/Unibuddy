# 🎓 UniBuddy MERN Authentication - Complete System Guide

## 📋 Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Features](#features)
4. [Technology Stack](#technology-stack)
5. [Setup & Installation](#setup--installation)
6. [Authentication Flow](#authentication-flow)
7. [Admin Panel Guide](#admin-panel-guide)
8. [API Documentation](#api-documentation)
9. [Security Features](#security-features)
10. [Troubleshooting](#troubleshooting)

---

## 🎯 System Overview

UniBuddy is a comprehensive MERN stack authentication system with Role-Based Access Control (RBAC), featuring a professional dark-themed UI with modal-based authentication forms and a full-featured admin panel for student management.

### Key Highlights
- ✅ JWT-based authentication with HTTP-only cookies
- ✅ Role-Based Access Control (RBAC) - Admin & User roles
- ✅ Modal-based authentication UI (Login, Signup, Forgot Password)
- ✅ Email verification with Mailtrap
- ✅ Password reset functionality
- ✅ Professional dark theme with Tailwind CSS
- ✅ Smooth animations with Framer Motion
- ✅ Full CRUD admin panel for student management
- ✅ Real-time statistics and filtering
- ✅ Export to CSV functionality

---

## 🏗️ Architecture

### Frontend Architecture
```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── AuthButton.jsx   # One-click auth modal trigger
│   │   ├── AuthModal.jsx    # Modal wrapper for auth forms
│   │   ├── Input.jsx        # Custom input component
│   │   └── ...
│   ├── pages/               # Page components
│   │   ├── LoginPage.jsx    # Login form (modal + direct)
│   │   ├── SignUpPage.jsx   # Signup form (modal + direct)
│   │   ├── DashboardPage.jsx
│   │   ├── AdminPanel.jsx   # Full CRUD admin panel
│   │   └── ...
│   ├── store/               # State management
│   │   └── authStore.js     # Zustand store for auth
│   └── context/             # React Context
│       └── AuthModalContext.jsx
```

### Backend Architecture
```
backend/
├── controllers/             # Business logic
│   ├── auth.controller.js   # Auth operations
│   └── user.controller.js   # User operations
├── models/                  # MongoDB schemas
│   ├── user.model.js        # User schema with RBAC
│   └── student.model.js     # Student schema
├── routes/                  # API routes
│   ├── auth.route.js
│   ├── user.routes.js
│   └── student.route.js
├── middleware/              # Custom middleware
│   └── verifyToken.js       # JWT verification
└── mailtrap/                # Email service
    ├── mailer.config.js
    ├── emails.js
    └── emailTemplates.js
```

---

## ✨ Features

### Authentication Features
1. **User Registration**
   - Email validation (@gdgu.org domain)
   - Password strength meter
   - Automatic email verification
   - Role assignment (USER by default)

2. **User Login**
   - Email/password authentication
   - JWT token generation
   - HTTP-only cookie storage
   - Remember me functionality

3. **Email Verification**
   - 6-digit verification code
   - Expiry time (24 hours)
   - Resend verification option
   - Beautiful email templates

4. **Password Reset**
   - Forgot password flow
   - Secure reset token
   - Email with reset link
   - Password strength validation

5. **Profile Management**
   - Update profile information
   - Change password
   - View account details

### Admin Panel Features
1. **Student Management**
   - Add new students
   - Edit student details
   - Delete students
   - View all students

2. **Search & Filter**
   - Search by name, email, roll no, department
   - Filter by status (Active, Inactive, Suspended)
   - Real-time filtering

3. **Statistics Dashboard**
   - Total students count
   - Active students
   - Inactive students
   - Suspended students

4. **Data Export**
   - Export to CSV
   - Filtered data export
   - Date-stamped files

5. **Pagination**
   - 10 students per page
   - Page navigation
   - Total count display

### UI/UX Features
1. **Modal-Based Authentication**
   - Popup forms instead of full pages
   - One-click triggers with AuthButton
   - Smooth animations
   - Close button inside forms

2. **Professional Dark Theme**
   - Slate gray background
   - Blue/purple/indigo accents
   - Gradient effects
   - Glass morphism

3. **Responsive Design**
   - Mobile-friendly
   - Tablet optimized
   - Desktop enhanced

4. **Animations**
   - Framer Motion transitions
   - Smooth page transitions
   - Loading states
   - Hover effects

---

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Zustand** - State management
- **React Router DOM** - Routing
- **Axios** - HTTP client
- **React Hot Toast** - Notifications
- **Lucide React** - Icons

### Backend
- **Node.js** - Runtime
- **Express.js** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM
- **JWT** - Authentication
- **bcryptjs** - Password hashing
- **Mailtrap** - Email service
- **cookie-parser** - Cookie handling
- **cors** - CORS handling
- **dotenv** - Environment variables

---

## 🚀 Setup & Installation

### Prerequisites
- Node.js (v14 or higher)
- MongoDB (local or Atlas)
- Mailtrap account (for emails)

### Environment Variables

Create `.env` file in root:
```env
# Server
PORT=5000
CLIENT_URL=http://localhost:5173

# Database
MONGO_URI=your_mongodb_connection_string

# JWT
JWT_SECRET=your_super_secret_jwt_key

# Mailtrap
MAILTRAP_TOKEN=your_mailtrap_token
MAILTRAP_ENDPOINT=https://send.api.mailtrap.io/

# Email
EMAIL_FROM=noreply@unibuddy.com
EMAIL_FROM_NAME=UniBuddy
```

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Unibuddy-MERN-Authentication
```

2. **Install backend dependencies**
```bash
npm install
```

3. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

4. **Start MongoDB**
```bash
# If using local MongoDB
mongod
```

5. **Start backend server**
```bash
npm run dev
```

6. **Start frontend server** (in new terminal)
```bash
cd frontend
npm run dev
```

7. **Access the application**
- Frontend: http://localhost:5173
- Backend: http://localhost:5000

---

## 🔐 Authentication Flow

### Registration Flow
1. User fills signup form
2. Frontend validates input (email format, password strength)
3. Backend receives request
4. Check if user already exists
5. Hash password with bcrypt
6. Generate verification token
7. Save user to database
8. Send verification email
9. Return JWT token
10. Set HTTP-only cookie
11. Redirect to email verification page

### Login Flow
1. User enters credentials
2. Frontend sends login request
3. Backend verifies email exists
4. Compare password with bcrypt
5. Check if email is verified
6. Generate JWT token
7. Set HTTP-only cookie
8. Return user data
9. Update Zustand store
10. Redirect based on role (Admin → Admin Panel, User → Dashboard)

### Email Verification Flow
1. User receives email with 6-digit code
2. User enters code on verification page
3. Backend validates code
4. Check expiry time
5. Mark email as verified
6. Update user in database
7. Return success message
8. Redirect to dashboard

### Password Reset Flow
1. User clicks "Forgot Password"
2. Enter email address
3. Backend generates reset token
4. Send reset email with link
5. User clicks link
6. Redirected to reset password page
7. Enter new password
8. Backend validates token
9. Hash new password
10. Update user in database
11. Send confirmation email
12. Redirect to login

---

## 👨‍💼 Admin Panel Guide

### Accessing Admin Panel
- Only users with `role: 'ADMIN'` can access
- Navigate to `/admin` route
- Protected by `verifyToken` and `isAdmin` middleware

### Admin Whitelist
Admin emails are defined in `backend/config/adminWhitelist.js`:
```javascript
export const adminEmails = [
  'admin@gdgu.org',
  'superadmin@gdgu.org'
];
```

### Student Management

#### Adding a Student
1. Click "Add Student" button
2. Fill required fields:
   - Name *
   - Email * (@gdgu.org)
   - Roll Number *
   - Department *
   - Year * (1-4)
   - Father's Name *
   - Mother's Name *
   - Contact Number * (10 digits)
3. Optional fields:
   - Date of Birth
   - Gender
   - Address
   - Photo URL
   - College ID Card URL
   - Status (Active/Inactive/Suspended)
4. Click "Add Student"

#### Editing a Student
1. Click edit icon on student row
2. Modify fields
3. Click "Update Student"

#### Deleting a Student
1. Click delete icon on student row
2. Confirm deletion in modal
3. Student is permanently removed

#### Search & Filter
- Search bar: Type name, email, roll no, or department
- Status filter: Select All/Active/Inactive/Suspended
- Results update in real-time

#### Export Data
- Click "Export" button
- Downloads CSV file with current filtered data
- Filename includes date: `students_2024-02-14.csv`

---

## 📡 API Documentation

### Authentication Endpoints

#### POST /api/auth/signup
Register a new user
```json
Request:
{
  "email": "user@gdgu.org",
  "password": "SecurePass123!",
  "name": "John Doe"
}

Response:
{
  "success": true,
  "message": "User created successfully",
  "user": { ... }
}
```

#### POST /api/auth/login
Login user
```json
Request:
{
  "email": "user@gdgu.org",
  "password": "SecurePass123!"
}

Response:
{
  "success": true,
  "message": "Logged in successfully",
  "user": { ... }
}
```

#### POST /api/auth/logout
Logout user
```json
Response:
{
  "success": true,
  "message": "Logged out successfully"
}
```

#### POST /api/auth/verify-email
Verify email with code
```json
Request:
{
  "code": "123456"
}

Response:
{
  "success": true,
  "message": "Email verified successfully"
}
```

#### POST /api/auth/forgot-password
Request password reset
```json
Request:
{
  "email": "user@gdgu.org"
}

Response:
{
  "success": true,
  "message": "Password reset link sent"
}
```

#### POST /api/auth/reset-password/:token
Reset password
```json
Request:
{
  "password": "NewSecurePass123!"
}

Response:
{
  "success": true,
  "message": "Password reset successful"
}
```

### Student Endpoints (Admin Only)

#### GET /api/students
Get all students
```json
Response:
[
  {
    "_id": "...",
    "name": "John Doe",
    "email": "john@gdgu.org",
    "rollNo": "2024001",
    "department": "Computer Science",
    "year": "3",
    "status": "ACTIVE",
    ...
  }
]
```

#### POST /api/students
Add new student
```json
Request:
{
  "name": "John Doe",
  "email": "john@gdgu.org",
  "rollNo": "2024001",
  "department": "Computer Science",
  "year": "3",
  "fatherName": "Father Name",
  "motherName": "Mother Name",
  "contactNumber": "1234567890"
}

Response:
{
  "message": "Student added successfully",
  "student": { ... }
}
```

#### PUT /api/students/:id
Update student
```json
Request:
{
  "name": "John Updated",
  "status": "INACTIVE"
}

Response:
{
  "message": "Student updated successfully",
  "student": { ... }
}
```

#### DELETE /api/students/:id
Delete student
```json
Response:
{
  "message": "Student deleted successfully"
}
```

---

## 🔒 Security Features

### 1. Password Security
- **bcrypt hashing**: Passwords hashed with salt rounds (10)
- **Password strength validation**: Minimum 8 characters, uppercase, lowercase, number, special char
- **No plain text storage**: Passwords never stored in plain text

### 2. JWT Security
- **HTTP-only cookies**: Tokens stored in HTTP-only cookies (not accessible via JavaScript)
- **Secure flag**: Cookies sent only over HTTPS in production
- **Expiry time**: Tokens expire after 7 days
- **Secret key**: Strong secret key for signing tokens

### 3. CSRF Protection
- **SameSite cookie attribute**: Prevents CSRF attacks
- **Origin validation**: CORS configured for specific origin

### 4. XSS Protection
- **Input sanitization**: All inputs sanitized before storage
- **Output encoding**: Data encoded before rendering
- **Content Security Policy**: CSP headers set

### 5. Email Validation
- **Domain restriction**: Only @gdgu.org emails allowed
- **Format validation**: Email format validated
- **Verification required**: Email must be verified before full access

### 6. Rate Limiting
- **Login attempts**: Limited login attempts per IP
- **API rate limiting**: Prevents brute force attacks

### 7. Role-Based Access Control (RBAC)
- **Role verification**: Middleware checks user role
- **Admin-only routes**: Protected routes for admin
- **Permission-based access**: Fine-grained access control

---

## 🐛 Troubleshooting

### Common Issues

#### 1. 500 Internal Server Error when adding student
**Problem**: Date of Birth field sending empty string instead of null

**Solution**: Updated backend to handle empty dateOfBirth:
```javascript
dateOfBirth: dateOfBirth && dateOfBirth.trim() !== "" ? dateOfBirth : undefined
```

#### 2. Close button not showing on modal
**Problem**: React Hooks called conditionally

**Solution**: Always call `useAuthModal()` at top level:
```javascript
const { isOpen, closeModal } = useAuthModal();
// Then use conditionally
{isOpen && <button onClick={closeModal}>Close</button>}
```

#### 3. Modal not centered
**Problem**: Fixed width on modal container

**Solution**: Remove `max-w-2xl` from AuthModal, let child components control width

#### 4. MongoDB connection failed
**Problem**: Invalid connection string or MongoDB not running

**Solution**:
- Check MONGO_URI in .env
- Ensure MongoDB is running
- Check network connectivity

#### 5. Email not sending
**Problem**: Invalid Mailtrap credentials

**Solution**:
- Verify MAILTRAP_TOKEN in .env
- Check Mailtrap account status
- Verify endpoint URL

#### 6. JWT token not working
**Problem**: Cookie not being sent with requests

**Solution**:
- Ensure `withCredentials: true` in axios requests
- Check CORS configuration
- Verify cookie settings (httpOnly, sameSite)

#### 7. Admin panel not accessible
**Problem**: User doesn't have admin role

**Solution**:
- Add email to adminWhitelist.js
- Manually update user role in database:
```javascript
db.users.updateOne(
  { email: "admin@gdgu.org" },
  { $set: { role: "ADMIN" } }
)
```

### Debug Mode

Enable detailed logging:
```javascript
// Backend
console.log("Request body:", req.body);
console.log("User:", req.user);

// Frontend
console.log("Form data:", form);
console.log("API response:", response.data);
```

---

## 📚 Additional Resources

- [RBAC Documentation](./RBAC-DOCUMENTATION.md)
- [Admin Panel Guide](./ADMIN-PANEL-GUIDE.md)
- [Interview Preparation](./INTERVIEW-PREP.md)

---

## 🎉 Conclusion

This system provides a complete, production-ready authentication solution with:
- ✅ Secure authentication flow
- ✅ Professional UI/UX
- ✅ Role-based access control
- ✅ Full admin panel
- ✅ Email verification
- ✅ Password reset
- ✅ Student management

Perfect for college management systems, educational platforms, or any application requiring robust authentication with admin capabilities.

---

**Built with ❤️ using MERN Stack**
