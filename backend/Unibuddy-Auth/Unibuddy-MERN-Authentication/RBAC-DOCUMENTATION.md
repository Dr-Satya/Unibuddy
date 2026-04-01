# 🔐 RBAC (Role-Based Access Control) Documentation

## ✅ Yes, RBAC is Fully Implemented!

Your application has a complete Role-Based Access Control system with two roles: **STUDENT** and **ADMIN**.

---

## 📋 Overview

### Roles Defined
1. **STUDENT** (Default Role)
   - Regular users who sign up
   - Access to student dashboard
   - Cannot access admin panel
   - Cannot manage other students

2. **ADMIN** (Privileged Role)
   - Assigned via email whitelist
   - Full access to admin panel
   - Can manage all students (CRUD operations)
   - Automatically redirected to admin panel on login

---

## 🏗️ Implementation Details

### 1. User Model (`backend/models/user.model.js`)

```javascript
role: {
  type: String,
  enum: ['STUDENT', 'ADMIN'],
  default: 'STUDENT'
}
```

- Role is stored in the database
- Only two valid roles: STUDENT and ADMIN
- Default role is STUDENT for new signups

### 2. Admin Whitelist (`backend/config/adminWhitelist.js`)

```javascript
export const ADMIN_WHITELIST = [
  'saafin@gdgu.org',
  '230160223057.saafin@gdgu.org',
  'samkit@gdgu.org',
  'kiyosha@gdgu.org'
];

export const isAdminEmail = (email) => {
  return ADMIN_WHITELIST.includes(email.toLowerCase());
};
```

**How it works:**
- Predefined list of admin emails
- During signup, email is checked against this list
- If email is in whitelist → role = 'ADMIN'
- If email is not in whitelist → role = 'STUDENT'

### 3. Authentication Middleware (`backend/middleware/verifyToken.js`)

```javascript
export const verifyToken = async (req, res, next) => {
  const token = req.cookies.token;
  
  // Verify JWT token
  const decoded = jwt.verify(token, process.env.JWT_SECRET);
  
  // Fetch user from database
  const user = await User.findById(decoded.userId).select("-password");
  
  // Attach user to request
  req.user = user;
  
  next();
};
```

**Features:**
- Verifies JWT token
- Fetches complete user object (including role)
- Attaches user to request for downstream middleware

### 4. Admin Authorization Middleware

**In Student Routes (`backend/routes/student.route.js`):**

```javascript
const isAdmin = (req, res, next) => {
  if (req.user && req.user.role === 'ADMIN') {
    next();
  } else {
    res.status(403).json({ 
      success: false, 
      message: 'Access denied. Admin only.' 
    });
  }
};
```

**Usage:**
```javascript
// All student management routes require admin role
router.get("/", verifyToken, isAdmin, async (req, res) => {...});
router.post("/", verifyToken, isAdmin, async (req, res) => {...});
router.put("/:id", verifyToken, isAdmin, async (req, res) => {...});
router.delete("/:id", verifyToken, isAdmin, async (req, res) => {...});
```

### 5. Frontend Route Protection (`frontend/src/App.jsx`)

**Protected Route (Students Only):**
```javascript
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return <LoadingSpinner />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!user?.isVerified) return <Navigate to="/verify-email" replace />;

  // Redirect admin to admin panel
  if (user?.role === 'ADMIN') return <Navigate to="/admin" replace />;

  return children;
};
```

**Admin Route (Admins Only):**
```javascript
const AdminRoute = ({ children }) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return <LoadingSpinner />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!user?.isVerified) return <Navigate to="/verify-email" replace />;
  
  // Only admins can access
  if (user?.role !== 'ADMIN') return <Navigate to="/" replace />;

  return children;
};
```

**Redirect Authenticated Users:**
```javascript
const RedirectAuthenticatedUser = ({ children }) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return <LoadingSpinner />;

  if (isAuthenticated && user?.isVerified) {
    // Redirect based on role
    if (user?.role === 'ADMIN') {
      return <Navigate to="/admin" replace />;
    }
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};
```

---

## 🔒 Access Control Matrix

| Route/Feature | Public | Student | Admin |
|--------------|--------|---------|-------|
| Landing Page (`/`) | ✅ | ❌ (redirects) | ❌ (redirects) |
| Demo Page (`/demo`) | ✅ | ❌ (redirects) | ❌ (redirects) |
| Login (`/login`) | ✅ | ❌ (redirects) | ❌ (redirects) |
| Signup (`/signup`) | ✅ | ❌ (redirects) | ❌ (redirects) |
| Email Verification | ✅ | ✅ | ✅ |
| Forgot Password | ✅ | ✅ | ✅ |
| Reset Password | ✅ | ✅ | ✅ |
| Student Dashboard (`/dashboard`) | ❌ | ✅ | ❌ (redirects to /admin) |
| Admin Panel (`/admin`) | ❌ | ❌ (403) | ✅ |
| View All Students (API) | ❌ | ❌ (403) | ✅ |
| Add Student (API) | ❌ | ❌ (403) | ✅ |
| Update Student (API) | ❌ | ❌ (403) | ✅ |
| Delete Student (API) | ❌ | ❌ (403) | ✅ |

---

## 🚀 How to Add a New Admin

### Method 1: Update Whitelist (Recommended)

1. Open `backend/config/adminWhitelist.js`
2. Add the email to the array:

```javascript
export const ADMIN_WHITELIST = [
  'saafin@gdgu.org',
  '230160223057.saafin@gdgu.org',
  'samkit@gdgu.org',
  'kiyosha@gdgu.org',
  'newemail@gdgu.org'  // ← Add new admin email here
];
```

3. Restart the backend server
4. User signs up with that email → automatically gets ADMIN role

### Method 2: Database Update (For Existing Users)

If a user already exists and you want to make them admin:

```javascript
// Using MongoDB shell or Compass
db.users.updateOne(
  { email: "user@gdgu.org" },
  { $set: { role: "ADMIN" } }
)
```

Or using Mongoose in Node.js:

```javascript
import { User } from "./models/user.model.js";

await User.updateOne(
  { email: "user@gdgu.org" },
  { role: "ADMIN" }
);
```

---

## 🔐 Security Features

### 1. Token-Based Authentication
- JWT tokens stored in HTTP-only cookies
- Tokens verified on every protected request
- Automatic token expiration

### 2. Role Verification
- Role checked on both frontend and backend
- Frontend: Route guards prevent unauthorized access
- Backend: Middleware blocks unauthorized API calls

### 3. Email Validation
- Only @gdgu.org emails allowed
- Admin emails must be in whitelist
- Email uniqueness enforced

### 4. Multi-Layer Protection
```
User Request
    ↓
Frontend Route Guard (checks role)
    ↓
Backend verifyToken (checks authentication)
    ↓
Backend isAdmin (checks authorization)
    ↓
Protected Resource
```

---

## 📊 User Flow Examples

### Student Signup Flow
```
1. User signs up with student@gdgu.org
2. Email NOT in admin whitelist
3. User created with role: 'STUDENT'
4. After verification → redirected to /dashboard
5. Cannot access /admin (403 error)
```

### Admin Signup Flow
```
1. User signs up with saafin@gdgu.org
2. Email IS in admin whitelist
3. User created with role: 'ADMIN'
4. After verification → redirected to /admin
5. Can manage all students
```

### Login Flow (Role-Based Redirect)
```
1. User logs in
2. System checks user.role
3. If ADMIN → redirect to /admin
4. If STUDENT → redirect to /dashboard
```

---

## 🧪 Testing RBAC

### Test as Student
1. Sign up with any @gdgu.org email NOT in whitelist
2. Verify email
3. Try to access `/admin` → Should redirect to `/dashboard`
4. Try API call to `/api/students` → Should get 403 error

### Test as Admin
1. Sign up with email from whitelist (e.g., `saafin@gdgu.org`)
2. Verify email
3. Should automatically redirect to `/admin`
4. Can view/add/edit/delete students
5. Try to access `/dashboard` → Should redirect to `/admin`

### Test API Protection
```bash
# Without admin role (should fail)
curl -X GET http://localhost:5000/api/students \
  -H "Cookie: token=student_token"
# Response: 403 Forbidden

# With admin role (should succeed)
curl -X GET http://localhost:5000/api/students \
  -H "Cookie: token=admin_token"
# Response: Array of students
```

---

## 🛠️ Customization Options

### Add More Roles

1. Update User Model:
```javascript
role: {
  type: String,
  enum: ['STUDENT', 'ADMIN', 'TEACHER', 'STAFF'],
  default: 'STUDENT'
}
```

2. Create Role-Specific Middleware:
```javascript
const isTeacher = (req, res, next) => {
  if (req.user && req.user.role === 'TEACHER') {
    next();
  } else {
    res.status(403).json({ message: 'Access denied. Teacher only.' });
  }
};
```

3. Add Frontend Route Guards:
```javascript
const TeacherRoute = ({ children }) => {
  const { user } = useAuthStore();
  if (user?.role !== 'TEACHER') return <Navigate to="/" />;
  return children;
};
```

### Permission-Based Access

For more granular control, implement permissions:

```javascript
// User Model
permissions: {
  type: [String],
  default: []
}

// Example permissions
const studentPermissions = ['view_own_profile', 'edit_own_profile'];
const adminPermissions = ['view_all_students', 'edit_all_students', 'delete_students'];

// Middleware
const hasPermission = (permission) => {
  return (req, res, next) => {
    if (req.user.permissions.includes(permission)) {
      next();
    } else {
      res.status(403).json({ message: 'Insufficient permissions' });
    }
  };
};

// Usage
router.delete("/:id", verifyToken, hasPermission('delete_students'), deleteStudent);
```

---

## 📝 Best Practices

1. **Always verify on backend** - Never trust frontend checks alone
2. **Use middleware chains** - `verifyToken` → `isAdmin` → `handler`
3. **Keep whitelist updated** - Regularly review admin list
4. **Log access attempts** - Track who tries to access admin features
5. **Use environment variables** - Store sensitive configs in .env
6. **Implement audit logs** - Track admin actions for accountability

---

## 🔍 Troubleshooting

### User not getting admin role
- Check if email is in `ADMIN_WHITELIST`
- Verify email is lowercase in whitelist
- Check database: `db.users.findOne({ email: "user@gdgu.org" })`

### 403 Forbidden errors
- Verify user is authenticated (has valid token)
- Check user role in database
- Ensure middleware order: `verifyToken` before `isAdmin`

### Admin redirected to student dashboard
- Check frontend route guards
- Verify `user.role` is correctly set
- Clear browser cache and cookies

---

## ✅ Summary

Your RBAC implementation includes:

- ✅ Two roles: STUDENT and ADMIN
- ✅ Email-based admin whitelist
- ✅ Automatic role assignment on signup
- ✅ Frontend route protection
- ✅ Backend API protection
- ✅ Role-based redirects
- ✅ Secure token verification
- ✅ Multi-layer security

**Your application is production-ready with proper RBAC!** 🎉
