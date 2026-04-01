# 🎯 Complete Authentication System - Interview Preparation Guide

## Part 1: Technologies & Stack

### Q26: What technologies did you use in this project?

**A:** Full MERN Stack with modern tools:

**Backend:**
- Node.js (Runtime environment)
- Express.js (Web framework)
- MongoDB (Database)
- Mongoose (ODM - Object Data Modeling)

**Frontend:**
- React.js (UI library)
- Vite (Build tool)
- Tailwind CSS (Styling)
- Framer Motion (Animations)

**Authentication:**
- JWT (jsonwebtoken)
- bcryptjs (Password hashing)
- cookie-parser (Cookie handling)

**State Management:**
- Zustand (Global state)
- React Router (Navigation)

**Email Service:**
- Mailtrap (Email testing)
- Nodemailer (Email sending)

**Additional:**
- Axios (HTTP client)
- React Hot Toast (Notifications)
- Lucide React (Icons)

---

### Q27: Why did you choose MERN stack?

**A:**

**MongoDB:**
- Flexible schema (easy to add fields)
- JSON-like documents (matches JavaScript)
- Scalable for growing data
- Good for user authentication systems

**Express.js:**
- Minimal and flexible
- Large ecosystem of middleware
- Easy to build REST APIs
- Industry standard

**React:**
- Component-based architecture
- Virtual DOM (fast updates)
- Large community and resources
- Modern and in-demand

**Node.js:**
- JavaScript everywhere (frontend + backend)
- Non-blocking I/O (handles many requests)
- NPM ecosystem (millions of packages)
- Fast and efficient

---

### Q28: What is Mongoose and why use it?

**A:** Mongoose is an ODM (Object Data Modeling) library for MongoDB.

**What it does:**
```javascript
// Without Mongoose (raw MongoDB)
db.collection('users').insertOne({
  email: "user@gdgu.org",
  password: "hash123"
});

// With Mongoose (structured)
const userSchema = new mongoose.Schema({
  email: { type: String, required: true },
  password: { type: String, required: true }
});

const User = mongoose.model('User', userSchema);
await User.create({ email, password });
```

**Benefits:**
- Schema validation (ensures data structure)
- Built-in type casting
- Query building helpers
- Middleware (pre/post hooks)
- Relationships between documents
- Input sanitization

**Example in your system:**
```javascript
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: true,
    unique: true,
    validate: {
      validator: (v) => v.endsWith('@gdgu.org'),
      message: 'Must be GDGU email'
    }
  }
});
```

---

### Q29: What is Express middleware?

**A:** Middleware are functions that execute during request-response cycle.

**Flow:**
```
Request → Middleware 1 → Middleware 2 → Route Handler → Response
```

**Types in your system:**

1. **Built-in middleware:**
```javascript
app.use(express.json()); // Parse JSON bodies
app.use(cookieParser()); // Parse cookies
```

2. **Third-party middleware:**
```javascript
app.use(cors({ origin: "http://localhost:5173" })); // CORS
```

3. **Custom middleware:**
```javascript
// Authentication middleware
export const verifyToken = async (req, res, next) => {
  const token = req.cookies.token;
  if (!token) return res.status(401).json({ message: "Unauthorized" });
  
  const decoded = jwt.verify(token, process.env.JWT_SECRET);
  req.user = await User.findById(decoded.userId);
  next(); // Pass to next middleware/route
};

// Usage
app.get("/api/students", verifyToken, getStudents);
```

4. **Error handling middleware:**
```javascript
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ message: "Server error" });
});
```

**Why middleware is powerful:**
- Reusable code (DRY principle)
- Separation of concerns
- Easy to add/remove features
- Clean route handlers

---

### Q30: What is Vite and why use it over Create React App?

**A:** Vite is a modern build tool for frontend development.

**Comparison:**

| Feature | Vite | Create React App |
|---------|------|------------------|
| **Dev Server Start** | ~100ms | ~10s |
| **Hot Module Reload** | Instant | 1-3s |
| **Build Tool** | esbuild + Rollup | Webpack |
| **Bundle Size** | Smaller | Larger |
| **Modern** | ✅ Latest | ⚠️ Older |

**Why Vite is faster:**
```
Create React App:
Bundle entire app → Start server → Show page
(Takes 10+ seconds)

Vite:
Start server → Load only needed files → Show page
(Takes <1 second)
```

**Benefits:**
- Lightning-fast dev server
- Instant hot module replacement
- Optimized production builds
- Native ES modules support
- Better developer experience

---

## Part 2: Architecture & Design Patterns

### Q31: Explain your project architecture

**A:** Three-tier architecture with clear separation:

```
┌─────────────────────────────────────────┐
│           FRONTEND (React)              │
│  - Components (UI)                      │
│  - Pages (Routes)                       │
│  - Store (Zustand - State)              │
│  - Context (Modal management)           │
└─────────────────────────────────────────┘
                  ↕ HTTP/HTTPS
┌─────────────────────────────────────────┐
│          BACKEND (Express)              │
│  - Routes (API endpoints)               │
│  - Controllers (Business logic)         │
│  - Middleware (Auth, validation)        │
│  - Models (Data structure)              │
└─────────────────────────────────────────┘
                  ↕ Mongoose
┌─────────────────────────────────────────┐
│         DATABASE (MongoDB)              │
│  - Users collection                     │
│  - Students collection                  │
└─────────────────────────────────────────┘
```

**Folder Structure:**
```
backend/
├── config/          # Configuration files
├── controllers/     # Business logic
├── middleware/      # Auth, validation
├── models/          # Database schemas
├── routes/          # API endpoints
├── mailtrap/        # Email service
└── index.js         # Entry point

frontend/
├── src/
│   ├── components/  # Reusable UI
│   ├── pages/       # Route pages
│   ├── store/       # Zustand store
│   ├── context/     # React context
│   └── utils/       # Helper functions
```

---

### Q32: What design patterns did you use?

**A:** Multiple design patterns for clean code:

**1. MVC Pattern (Backend):**
```javascript
// Model (Data)
const User = mongoose.model('User', userSchema);

// Controller (Logic)
export const login = async (req, res) => {
  const user = await User.findOne({ email });
  // Business logic here
};

// Route (Entry point)
router.post('/login', login);
```

**2. Middleware Pattern:**
```javascript
// Chain of responsibility
router.delete('/students/:id', 
  verifyToken,    // Check authentication
  isAdmin,        // Check authorization
  deleteStudent   // Execute action
);
```

**3. Repository Pattern:**
```javascript
// Abstraction over database
class UserRepository {
  async findByEmail(email) {
    return await User.findOne({ email });
  }
  
  async create(userData) {
    return await User.create(userData);
  }
}
```

**4. Factory Pattern:**
```javascript
// Create different types of users
function createUser(type, data) {
  if (type === 'ADMIN') {
    return new Admin(data);
  }
  return new Student(data);
}
```

**5. Observer Pattern (React):**
```javascript
// Zustand store notifies components of changes
const { user } = useAuthStore(); // Subscribes to changes
```

---

### Q33: How did you structure your API?

**A:** RESTful API with clear conventions:

**Endpoints:**
```
Authentication:
POST   /api/auth/signup           # Create account
POST   /api/auth/login            # Login
POST   /api/auth/logout           # Logout
GET    /api/auth/check-auth       # Verify session
POST   /api/auth/verify-email     # Verify email
POST   /api/auth/forgot-password  # Request reset
POST   /api/auth/reset-password   # Reset password

Students (Admin only):
GET    /api/students              # Get all
GET    /api/students/:id          # Get one
POST   /api/students              # Create
PUT    /api/students/:id          # Update
DELETE /api/students/:id          # Delete

Users (Admin only):
GET    /api/users                 # Get all users
POST   /api/users                 # Create user
PUT    /api/users/:id             # Update user
DELETE /api/users/:id             # Delete user
```

**REST Principles:**
- Use HTTP methods correctly (GET, POST, PUT, DELETE)
- Resource-based URLs (/students not /getStudents)
- Stateless (no server-side sessions)
- Standard status codes (200, 201, 400, 401, 403, 500)

**Response Format:**
```javascript
// Success
{
  success: true,
  message: "Student created",
  data: { student }
}

// Error
{
  success: false,
  message: "Validation failed",
  errors: [...]
}
```

---

### Q34: How do you handle errors?

**A:** Comprehensive error handling at multiple levels:

**1. Try-Catch Blocks:**
```javascript
export const login = async (req, res) => {
  try {
    const user = await User.findOne({ email });
    // Logic here
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).json({ message: "Server error" });
  }
};
```

**2. Mongoose Validation:**
```javascript
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    required: [true, "Email is required"],
    validate: {
      validator: (v) => v.endsWith('@gdgu.org'),
      message: "Must be GDGU email"
    }
  }
});
```

**3. Custom Error Classes:**
```javascript
class ValidationError extends Error {
  constructor(message) {
    super(message);
    this.statusCode = 400;
  }
}

throw new ValidationError("Invalid email");
```

**4. Global Error Handler:**
```javascript
app.use((err, req, res, next) => {
  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    success: false,
    message: err.message,
    stack: process.env.NODE_ENV === 'development' ? err.stack : undefined
  });
});
```

**5. Frontend Error Handling:**
```javascript
try {
  await axios.post('/api/auth/login', { email, password });
  toast.success("Login successful");
} catch (error) {
  const message = error.response?.data?.message || "Login failed";
  toast.error(message);
}
```

---

## Part 3: Database & Models

### Q35: Explain your database schema

**A:** Two main collections with relationships:

**User Schema:**
```javascript
{
  _id: ObjectId,
  email: String (unique, @gdgu.org),
  password: String (hashed),
  role: String (STUDENT/ADMIN),
  fatherName: String,
  motherName: String,
  contactNumber: String (10 digits),
  photo: String (base64/URL),
  collegeIdCard: String (base64/URL),
  isVerified: Boolean,
  verificationToken: String,
  verificationTokenExpiresAt: Date,
  resetPasswordToken: String,
  resetPasswordExpiresAt: Date,
  lastLogin: Date,
  createdAt: Date,
  updatedAt: Date
}
```

**Student Schema:**
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
  gender: String (Male/Female/Other),
  photo: String,
  collegeIdCard: String,
  status: String (ACTIVE/INACTIVE/SUSPENDED),
  createdAt: Date,
  updatedAt: Date
}
```

**Why separate collections:**
- Users = Authentication data
- Students = Academic/profile data
- Separation of concerns
- Different access patterns

---

### Q36: What are Mongoose indexes and why use them?

**A:** Indexes improve query performance.

**In your schema:**
```javascript
const userSchema = new mongoose.Schema({
  email: {
    type: String,
    unique: true  // Creates index automatically
  }
});

// Manual index
userSchema.index({ email: 1, role: 1 });
```

**How indexes work:**
```
Without index:
Find user by email → Scan all 10,000 users → O(n)

With index:
Find user by email → Binary search → O(log n)
```

**Trade-offs:**
- ✅ Faster reads
- ❌ Slower writes (must update index)
- ❌ More storage space

**When to use:**
- Fields used in queries (email, rollNo)
- Unique constraints (email, rollNo)
- Sorting fields (createdAt)

---

### Q37: What is the difference between SQL and NoSQL?

**A:** Different database paradigms:

| Feature | SQL (MySQL) | NoSQL (MongoDB) |
|---------|-------------|-----------------|
| **Structure** | Tables, rows | Collections, documents |
| **Schema** | Fixed | Flexible |
| **Relationships** | Foreign keys | Embedded/referenced |
| **Scaling** | Vertical | Horizontal |
| **Transactions** | ACID | Eventually consistent |
| **Query Language** | SQL | JavaScript-like |

**Example:**

**SQL:**
```sql
CREATE TABLE users (
  id INT PRIMARY KEY,
  email VARCHAR(255),
  password VARCHAR(255)
);

SELECT * FROM users WHERE email = 'user@gdgu.org';
```

**NoSQL (MongoDB):**
```javascript
// No schema required
db.users.insertOne({
  email: "user@gdgu.org",
  password: "hash",
  newField: "can add anytime"
});

db.users.findOne({ email: "user@gdgu.org" });
```

**Why MongoDB for this project:**
- Flexible schema (easy to add fields)
- JSON-like documents (natural for JavaScript)
- Fast development (no migrations)
- Good for user profiles (varying data)

---

### Q38: How do you handle database connections?

**A:** Connection pooling with error handling:

```javascript
// backend/db/connectDB.js
export const connectDB = async () => {
  try {
    const conn = await mongoose.connect(process.env.MONGO_URI, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    
    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.error("MongoDB connection error:", error);
    process.exit(1); // Exit if cannot connect
  }
};

// In index.js
connectDB();
```

**Connection pooling:**
- Mongoose maintains pool of connections
- Reuses connections (faster than creating new)
- Handles reconnection automatically
- Default pool size: 5

**Best practices:**
- Use environment variables for URI
- Handle connection errors
- Close connections on app shutdown
- Monitor connection health

---

## Part 4: Frontend Architecture

### Q39: Explain React component hierarchy

**A:** Component tree structure:

```
App (Root)
├── AuthModalProvider (Context)
│   ├── Routes
│   │   ├── LandingPage
│   │   │   └── AuthButton
│   │   ├── LoginPage
│   │   │   ├── Input
│   │   │   └── Form
│   │   ├── SignUpPage
│   │   │   ├── Input
│   │   │   ├── PasswordStrengthMeter
│   │   │   └── Form
│   │   ├── DashboardPage (Protected)
│   │   └── AdminPanel (Admin Only)
│   │       ├── Statistics Cards
│   │       ├── Search/Filter
│   │       ├── Table
│   │       └── Modals
│   └── AuthModalManager
│       └── AuthModal
│           └── Dynamic Content
└── Toaster (Notifications)
```

**Component types:**

1. **Page Components:** Full pages (LoginPage, DashboardPage)
2. **Layout Components:** Structure (AuthModal)
3. **UI Components:** Reusable (Input, Button)
4. **Feature Components:** Specific functionality (PasswordStrengthMeter)

---

### Q40: What is React Context and when to use it?

**A:** Context provides global state without prop drilling.

**Problem (Prop Drilling):**
```javascript
<App>
  <Header user={user} />
    <Navbar user={user} />
      <UserMenu user={user} /> // Passed through 3 levels!
```

**Solution (Context):**
```javascript
// Create context
const AuthModalContext = createContext();

// Provider
<AuthModalProvider>
  <App />
</AuthModalProvider>

// Use anywhere
const { openModal } = useAuthModal();
```

**In your system:**
```javascript
// AuthModalContext.jsx
export const AuthModalProvider = ({ children }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalType, setModalType] = useState(null);

  const openModal = (type) => {
    setModalType(type);
    setIsModalOpen(true);
  };

  return (
    <AuthModalContext.Provider value={{ isModalOpen, modalType, openModal }}>
      {children}
    </AuthModalContext.Provider>
  );
};

// Usage in any component
const { openModal } = useAuthModal();
<button onClick={() => openModal('login')}>Login</button>
```

**When to use Context:**
- ✅ Theme (dark/light mode)
- ✅ Authentication state
- ✅ Modal management
- ✅ Language/i18n
- ❌ Frequently changing data (use Zustand)
- ❌ Local component state

---

### Q41: What is Zustand and how does it work?

**A:** Zustand is a lightweight state management library.

**Why Zustand over Redux:**
- Less boilerplate (no actions, reducers)
- Simpler API
- Better TypeScript support
- Smaller bundle size
- No provider needed

**Implementation:**
```javascript
// store/authStore.js
export const useAuthStore = create((set) => ({
  // State
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,

  // Actions
  login: async (email, password) => {
    set({ isLoading: true, error: null });
    try {
      const response = await axios.post("/api/auth/login", { email, password });
      set({ 
        user: response.data.user, 
        isAuthenticated: true, 
        isLoading: false 
      });
    } catch (error) {
      set({ 
        error: error.response?.data?.message, 
        isLoading: false 
      });
    }
  },

  logout: async () => {
    await axios.post("/api/auth/logout");
    set({ user: null, isAuthenticated: false });
  },

  checkAuth: async () => {
    try {
      const response = await axios.get("/api/auth/check-auth");
      set({ user: response.data.user, isAuthenticated: true });
    } catch (error) {
      set({ user: null, isAuthenticated: false });
    }
  }
}));

// Usage in components
function Dashboard() {
  const { user, logout } = useAuthStore();
  
  return (
    <div>
      <h1>Welcome {user.name}</h1>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

**Benefits:**
- Components auto-update when state changes
- No prop drilling
- Easy to test
- DevTools support

---

### Q42: What is React Router and how did you use it?

**A:** React Router handles navigation in single-page apps.

**Key concepts:**

**1. Routes:**
```javascript
<Routes>
  <Route path="/" element={<LandingPage />} />
  <Route path="/login" element={<LoginPage />} />
  <Route path="/dashboard" element={<DashboardPage />} />
</Routes>
```

**2. Protected Routes:**
```javascript
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated } = useAuthStore();
  
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  
  return children;
};

<Route path="/dashboard" element={
  <ProtectedRoute>
    <DashboardPage />
  </ProtectedRoute>
} />
```

**3. Navigation:**
```javascript
// Declarative
<Link to="/dashboard">Dashboard</Link>

// Programmatic
const navigate = useNavigate();
navigate("/dashboard");
```

**4. URL Parameters:**
```javascript
<Route path="/reset-password/:token" element={<ResetPasswordPage />} />

// In component
const { token } = useParams();
```

**Why React Router:**
- Client-side routing (no page reload)
- Browser history management
- Nested routes support
- Protected routes
- URL parameters

---

### Q43: What are React Hooks and which ones did you use?

**A:** Hooks let you use state and lifecycle in function components.

**Hooks used in your project:**

**1. useState - Local state:**
```javascript
const [email, setEmail] = useState("");
const [loading, setLoading] = useState(false);
```

**2. useEffect - Side effects:**
```javascript
useEffect(() => {
  checkAuth(); // Run on mount
}, []);

useEffect(() => {
  fetchStudents(); // Run when dependency changes
}, [searchTerm]);
```

**3. useNavigate - Navigation:**
```javascript
const navigate = useNavigate();
navigate("/dashboard");
```

**4. useParams - URL parameters:**
```javascript
const { token } = useParams();
```

**5. useContext - Context access:**
```javascript
const { openModal } = useAuthModal();
```

**6. Custom hooks:**
```javascript
// useAuthStore (Zustand)
const { user, login, logout } = useAuthStore();

// useAuthModal (Context)
const { openModal, closeModal } = useAuthModal();
```

**Rules of Hooks:**
- ✅ Call at top level (not in loops/conditions)
- ✅ Call from React functions only
- ✅ Use "use" prefix for custom hooks

---

## Part 5: Styling & UI

### Q44: What is Tailwind CSS and why use it?

**A:** Tailwind is a utility-first CSS framework.

**Traditional CSS:**
```css
.button {
  background-color: blue;
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
}
```

**Tailwind:**
```jsx
<button className="bg-blue-600 text-white px-4 py-2 rounded-lg">
  Click me
</button>
```

**Benefits:**
- No naming classes (no "button-primary-large-blue")
- Faster development
- Smaller CSS bundle (unused styles purged)
- Consistent design system
- Responsive utilities built-in

**Responsive design:**
```jsx
<div className="
  w-full          // Mobile: full width
  md:w-1/2        // Tablet: half width
  lg:w-1/3        // Desktop: third width
">
```

**Why Tailwind for this project:**
- Rapid prototyping
- Consistent spacing/colors
- Easy to maintain
- No CSS file management
- Great with component libraries

---

### Q45: What is Framer Motion?

**A:** Framer Motion is an animation library for React.

**Usage in your project:**

**1. Page transitions:**
```javascript
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.5 }}
>
  <LoginPage />
</motion.div>
```

**2. Modal animations:**
```javascript
<AnimatePresence>
  {showModal && (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
    >
      <Modal />
    </motion.div>
  )}
</AnimatePresence>
```

**3. Button interactions:**
```javascript
<motion.button
  whileHover={{ scale: 1.05 }}
  whileTap={{ scale: 0.95 }}
>
  Click me
</motion.button>
```

**Why use animations:**
- Better user experience
- Visual feedback
- Professional appearance
- Smooth transitions
- Guides user attention

---

### Q46: How did you make the UI responsive?

**A:** Mobile-first approach with Tailwind breakpoints:

**Breakpoints:**
```
sm: 640px   (Mobile landscape)
md: 768px   (Tablet)
lg: 1024px  (Desktop)
xl: 1280px  (Large desktop)
```

**Example:**
```jsx
<div className="
  grid 
  grid-cols-1        // Mobile: 1 column
  md:grid-cols-2     // Tablet: 2 columns
  lg:grid-cols-4     // Desktop: 4 columns
  gap-4
">
  <StatCard />
  <StatCard />
  <StatCard />
  <StatCard />
</div>
```

**Admin Panel responsive:**
```jsx
// Controls bar
<div className="
  flex 
  flex-col           // Mobile: stack vertically
  md:flex-row        // Desktop: horizontal
  gap-4
">
  <SearchBar />
  <FilterDropdown />
  <ActionButtons />
</div>
```

**Testing responsive:**
- Chrome DevTools (F12 → Toggle device toolbar)
- Test on actual devices
- Check all breakpoints
- Ensure touch targets are large enough (44x44px minimum)

---

## Part 6: Email Service

### Q47: How does email sending work?

**A:** Using Nodemailer with Mailtrap for testing:

**Setup:**
```javascript
// mailtrap/mailer.config.js
import nodemailer from "nodemailer";

export const mailtrapClient = nodemailer.createTransport({
  host: process.env.MAILTRAP_HOST,
  port: process.env.MAILTRAP_PORT,
  auth: {
    user: process.env.MAILTRAP_USER,
    pass: process.env.MAILTRAP_PASS,
  },
});

export const sender = {
  email: "noreply@unibuddy.com",
  name: "UniBuddy Team",
};
```

**Sending emails:**
```javascript
// mailtrap/emails.js
export const sendVerificationEmail = async (email, verificationToken) => {
  const recipient = [{ email }];

  try {
    const response = await mailtrapClient.sendMail({
      from: sender,
      to: recipient,
      subject: "Verify your email",
      html: VERIFICATION_EMAIL_TEMPLATE.replace("{verificationCode}", verificationToken),
      category: "Email Verification",
    });

    console.log("Email sent successfully", response);
  } catch (error) {
    console.error("Error sending email", error);
    throw new Error(`Error sending email: ${error}`);
  }
};
```

**Email templates:**
```javascript
export const VERIFICATION_EMAIL_TEMPLATE = `
<!DOCTYPE html>
<html>
<body>
  <h1>Verify Your Email</h1>
  <p>Your verification code is:</p>
  <h2>{verificationCode}</h2>
  <p>This code expires in 24 hours.</p>
</body>
</html>
`;
```

**Why Mailtrap:**
- Safe testing (emails don't go to real users)
- Inspect email content
- Test spam score
- Free for development
- Easy to switch to production (SendGrid, AWS SES)

---

### Q48: What emails does your system send?

**A:** Four types of emails:

**1. Welcome Email:**
- Sent after signup
- Includes verification code
- Welcomes user to platform

**2. Verification Email:**
- Contains 6-digit code
- Expires in 24 hours
- Required to access account

**3. Password Reset Email:**
- Contains reset link with token
- Expires in 1 hour
- One-time use

**4. Success Email:**
- Sent after password reset
- Confirms password changed
- Security notification

**Email best practices:**
- Clear subject lines
- Mobile-responsive HTML
- Plain text fallback
- Unsubscribe link (for marketing)
- Sender authentication (SPF, DKIM)

---

## Part 7: Deployment & Production

### Q49: How would you deploy this application?

**A:** Modern deployment strategy:

**Backend (Node.js/Express):**

**Option 1: Heroku**
```bash
# Install Heroku CLI
heroku create unibuddy-api

# Set environment variables
heroku config:set MONGO_URI=mongodb://...
heroku config:set JWT_SECRET=...

# Deploy
git push heroku main
```

**Option 2: Railway/Render**
- Connect GitHub repo
- Auto-deploy on push
- Set environment variables
- Free tier available

**Option 3: AWS EC2**
- More control
- Requires server management
- Use PM2 for process management

**Frontend (React):**

**Option 1: Vercel** (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Production
vercel --prod
```

**Option 2: Netlify**
- Drag and drop build folder
- Or connect GitHub
- Auto-deploy on push

**Option 3: AWS S3 + CloudFront**
- Static hosting
- CDN for fast delivery
- More complex setup

**Database (MongoDB):**

**MongoDB Atlas** (Cloud)
- Free tier (512MB)
- Automatic backups
- Global clusters
- Easy to scale

**Deployment checklist:**
- ✅ Environment variables set
- ✅ CORS configured for production domain
- ✅ HTTPS enabled
- ✅ Database indexes created
- ✅ Error logging (Sentry)
- ✅ Performance monitoring
- ✅ Backup strategy

---

### Q50: What environment variables do you need?

**A:** All sensitive configuration:

**Backend (.env):**
```bash
# Server
PORT=5000
NODE_ENV=production

# Database
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/unibuddy

# JWT
JWT_SECRET=your-super-secret-key-here

# Email (Mailtrap)
MAILTRAP_HOST=smtp.mailtrap.io
MAILTRAP_PORT=2525
MAILTRAP_USER=your-username
MAILTRAP_PASS=your-password

# Frontend URL (CORS)
CLIENT_URL=https://unibuddy.vercel.app
```

**Frontend (.env):**
```bash
# API URL
VITE_API_URL=https://unibuddy-api.herokuapp.com
```

**Security:**
- Never commit .env to git
- Use .env.example for documentation
- Different values for dev/prod
- Rotate secrets regularly
- Use secret management (AWS Secrets Manager)

---

### Q51: How do you handle CORS in production?

**A:** Configure CORS for specific origins:

**Development:**
```javascript
app.use(cors({
  origin: "http://localhost:5173",
  credentials: true
}));
```

**Production:**
```javascript
const allowedOrigins = [
  "https://unibuddy.vercel.app",
  "https://www.unibuddy.com"
];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true
}));
```

**Why CORS:**
- Browser security feature
- Prevents unauthorized API access
- Protects against CSRF
- Allows specific domains only

---

### Q52: How do you monitor and debug production issues?

**A:** Multiple monitoring strategies:

**1. Error Logging (Sentry):**
```javascript
import * as Sentry from "@sentry/node";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
});

// Catch errors
app.use(Sentry.Handlers.errorHandler());
```

**2. Application Logs:**
```javascript
// Use Winston or Pino
import winston from "winston";

const logger = winston.createLogger({
  level: "info",
  format: winston.format.json(),
  transports: [
    new winston.transports.File({ filename: "error.log", level: "error" }),
    new winston.transports.File({ filename: "combined.log" }),
  ],
});

logger.error("Login failed", { email, error });
```

**3. Performance Monitoring:**
```javascript
// Response time middleware
app.use((req, res, next) => {
  const start = Date.now();
  res.on("finish", () => {
    const duration = Date.now() - start;
    console.log(`${req.method} ${req.url} - ${duration}ms`);
  });
  next();
});
```

**4. Database Monitoring:**
- MongoDB Atlas monitoring
- Query performance
- Connection pool stats
- Slow query logs

**5. Uptime Monitoring:**
- UptimeRobot (free)
- Pingdom
- StatusCake
- Alerts via email/SMS

---

## Part 8: Testing & Quality

### Q53: How would you test this application?

**A:** Comprehensive testing strategy:

**1. Unit Tests (Jest):**
```javascript
// Test password hashing
describe("Password hashing", () => {
  it("should hash password correctly", async () => {
    const password = "test123";
    const hashed = await bcrypt.hash(password, 10);
    expect(hashed).not.toBe(password);
    
    const isMatch = await bcrypt.compare(password, hashed);
    expect(isMatch).toBe(true);
  });
});

// Test JWT
describe("JWT", () => {
  it("should create and verify token", () => {
    const payload = { userId: "123" };
    const token = jwt.sign(payload, "secret");
    const decoded = jwt.verify(token, "secret");
    expect(decoded.userId).toBe("123");
  });
});
```

**2. Integration Tests (Supertest):**
```javascript
import request from "supertest";
import app from "../index.js";

describe("Auth API", () => {
  it("should signup new user", async () => {
    const response = await request(app)
      .post("/api/auth/signup")
      .send({
        email: "test@gdgu.org",
        password: "password123",
        fatherName: "Father",
        motherName: "Mother",
        contactNumber: "1234567890"
      });
    
    expect(response.status).toBe(201);
    expect(response.body.user.email).toBe("test@gdgu.org");
  });

  it("should login user", async () => {
    const response = await request(app)
      .post("/api/auth/login")
      .send({
        email: "test@gdgu.org",
        password: "password123"
      });
    
    expect(response.status).toBe(200);
    expect(response.body.user).toBeDefined();
  });
});
```

**3. Frontend Tests (React Testing Library):**
```javascript
import { render, screen, fireEvent } from "@testing-library/react";
import LoginPage from "./LoginPage";

describe("LoginPage", () => {
  it("should render login form", () => {
    render(<LoginPage />);
    expect(screen.getByPlaceholderText("Email")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Password")).toBeInTheDocument();
  });

  it("should show error for invalid email", async () => {
    render(<LoginPage />);
    const emailInput = screen.getByPlaceholderText("Email");
    fireEvent.change(emailInput, { target: { value: "invalid" } });
    fireEvent.blur(emailInput);
    
    expect(await screen.findByText("Must be GDGU email")).toBeInTheDocument();
  });
});
```

**4. E2E Tests (Cypress):**
```javascript
describe("Authentication Flow", () => {
  it("should complete signup flow", () => {
    cy.visit("/signup");
    cy.get('input[name="email"]').type("test@gdgu.org");
    cy.get('input[name="password"]').type("password123");
    cy.get('button[type="submit"]').click();
    
    cy.url().should("include", "/verify-email");
  });
});
```

**Test coverage goals:**
- Unit tests: 80%+
- Integration tests: Critical paths
- E2E tests: Main user flows

---
