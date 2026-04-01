import { useState } from "react";
import { motion } from "framer-motion";
import { Mail, Lock, Loader, X } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import Input from "../components/Input";
import { useAuthStore } from "../store/authStore";
import { useAuthModal } from "../context/AuthModalContext";

const LoginPage = ({ isModal = false }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);

  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);
  const isLoading = useAuthStore((state) => state.isLoading);
  
  // Always call the hook, but only use it when isModal is true
  const modalContext = useAuthModal();
  const closeModal = isModal ? modalContext.closeModal : () => {};
  const switchModal = isModal ? modalContext.switchModal : () => {};

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(null);
    try {
      await login(email, password);
      if (isModal) {
        closeModal(); // Close modal on successful login
      } else {
        navigate("/"); // redirect to dashboard after login
      }
    } catch (err) {
      setError(err.response?.data?.message || "Login failed");
    }
  };

  const handleSignupClick = (e) => {
    if (isModal) {
      e.preventDefault();
      switchModal('signup');
    }
  };

  const handleForgotPasswordClick = (e) => {
    if (isModal) {
      e.preventDefault();
      switchModal('forgot-password');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="max-w-md w-full bg-slate-900/95 backdrop-filter backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-800 overflow-hidden relative"
    >
      {/* Close Button - Only show in modal mode */}
      {isModal && (
        <button
          onClick={closeModal}
          className="absolute top-4 right-4 z-10 bg-slate-800/90 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full p-2 transition-all duration-200 hover:scale-110"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>
      )}

      <div className="p-8">
        <h2 className="text-3xl font-bold mb-2 text-center bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 text-transparent bg-clip-text">
          Welcome Back
        </h2>
        <p className="text-slate-400 text-center text-sm mb-8">Sign in to continue to your account</p>

        <form onSubmit={handleLogin}>
          <Input
            icon={Mail}
            type="email"
            placeholder="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <Input
            icon={Lock}
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <div className="flex items-center mb-6">
            {isModal ? (
              <button
                type="button"
                onClick={handleForgotPasswordClick}
                className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
              >
                Forgot password?
              </button>
            ) : (
              <Link
                to="/forgot-password"
                className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
              >
                Forgot password?
              </Link>
            )}
          </div>

          {error && <p className="text-red-400 text-sm font-medium mb-4 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{error}</p>}

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 transition duration-200"
            type="submit"
            disabled={isLoading}
          >
            {isLoading ? <Loader className="w-6 h-6 animate-spin mx-auto" /> : "Sign In"}
          </motion.button>
        </form>
      </div>
      <div className="px-8 py-4 bg-slate-950/50 border-t border-slate-800 flex justify-center">
        <p className="text-sm text-slate-400">
          Don't have an account?{" "}
          {isModal ? (
            <button
              onClick={handleSignupClick}
              className="text-blue-400 hover:text-blue-300 font-medium transition-colors"
            >
              Sign up
            </button>
          ) : (
            <Link to="/signup" className="text-blue-400 hover:text-blue-300 font-medium transition-colors">
              Sign up
            </Link>
          )}
        </p>
      </div>
    </motion.div>
  );
};

export default LoginPage;