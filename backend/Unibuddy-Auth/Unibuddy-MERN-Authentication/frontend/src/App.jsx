import { Navigate, Route, Routes } from "react-router-dom";
import { useEffect } from "react";
import { Toaster } from "react-hot-toast";

import FloatingShape from "./components/FloatingShape";
import LoadingSpinner from "./components/LoadingSpinner";

import SignUpPage from "./pages/SignUpPage";
import LoginPage from "./pages/LoginPage";
import EmailVerificationPage from "./pages/EmailVerificationPage";
import DashboardPage from "./pages/DashboardPage";
import AdminPanel from "./pages/AdminPanel";
import ForgotPasswordPage from "./pages/ForgotPasswordPage";
import ResetPasswordPage from "./pages/ResetPasswordPage";
import LandingPage from "./pages/LandingPage";
import DemoPage from "./pages/DemoPage";

import { useAuthStore } from "./store/authStore";
import { AuthModalProvider } from "./context/AuthModalContext";
import AuthModalManager from "./components/AuthModalManager";

// 🔐 Protected Route for Students
const ProtectedRoute = ({ children }) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return <LoadingSpinner />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!user?.isVerified) return <Navigate to="/verify-email" replace />;

  // Redirect admin to admin panel
  if (user?.role === 'ADMIN') return <Navigate to="/admin" replace />;

  return children;
};

// 🔐 Admin Only Route
const AdminRoute = ({ children }) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return <LoadingSpinner />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (!user?.isVerified) return <Navigate to="/verify-email" replace />;
  
  // Only admins can access
  if (user?.role !== 'ADMIN') return <Navigate to="/" replace />;

  return children;
};

// 🔁 Redirect authenticated users
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

function App() {
  const checkAuth = useAuthStore((state) => state.checkAuth);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  return (
    <AuthModalProvider>
      <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 flex items-center justify-center relative overflow-hidden">
        <FloatingShape color="bg-blue-600/20" size="w-64 h-64" top="-5%" left="10%" delay={0} />
        <FloatingShape color="bg-purple-600/20" size="w-48 h-48" top="70%" left="80%" delay={5} />
        <FloatingShape color="bg-indigo-600/20" size="w-32 h-32" top="40%" left="-10%" delay={2} />

        <Routes>
          {/* Landing Page - Public */}
          <Route 
            path="/" 
            element={
              <RedirectAuthenticatedUser>
                <LandingPage />
              </RedirectAuthenticatedUser>
            } 
          />

          {/* Demo Page - Public (shows the new UI features) */}
          <Route 
            path="/demo" 
            element={
              <RedirectAuthenticatedUser>
                <DemoPage />
              </RedirectAuthenticatedUser>
            } 
          />

          {/* Student Dashboard */}
          <Route 
            path="/dashboard" 
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            } 
          />

          {/* Admin Panel */}
          <Route 
            path="/admin" 
            element={
              <AdminRoute>
                <AdminPanel />
              </AdminRoute>
            } 
          />

          {/* Auth Routes - Keep for direct URL access */}
          <Route 
            path="/signup" 
            element={
              <RedirectAuthenticatedUser>
                <SignUpPage isModal={false} />
              </RedirectAuthenticatedUser>
            } 
          />
          <Route 
            path="/login" 
            element={
              <RedirectAuthenticatedUser>
                <LoginPage isModal={false} />
              </RedirectAuthenticatedUser>
            } 
          />

          <Route path="/verify-email" element={<EmailVerificationPage />} />
          <Route 
            path="/forgot-password" 
            element={
              <ForgotPasswordPage isModal={false} />
            } 
          />
          <Route path="/reset-password/:token" element={<ResetPasswordPage />} />

          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>

        {/* Modal Manager - Renders modals globally */}
        <AuthModalManager />

        <Toaster />
      </div>
    </AuthModalProvider>
  );
}

export default App;