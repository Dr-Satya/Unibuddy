import { motion } from "framer-motion";
import { useState } from "react";
import { useAuthStore } from "../store/authStore";
import Input from "../components/Input";
import { ArrowLeft, Loader, Mail, X } from "lucide-react";
import { Link } from "react-router-dom";
import { useAuthModal } from "../context/AuthModalContext";

const ForgotPasswordPage = ({ isModal = false }) => {
  const [email, setEmail] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  // ✅ Correct way to access Zustand state/functions
  const forgotPassword = useAuthStore((state) => state.forgotPassword);
  const isLoading = useAuthStore((state) => state.isLoading);
  
  // Always call the hook, but only use it when isModal is true
  const modalContext = useAuthModal();
  const switchModal = isModal ? modalContext.switchModal : () => {};
  const closeModal = isModal ? modalContext.closeModal : () => {};

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccessMsg("");

    try {
      await forgotPassword(email); // calls the store function
      setIsSubmitted(true);
      setSuccessMsg("Password reset link sent! Check your email.");
    } catch (err) {
      setError(err.response?.data?.message || "Something went wrong. Please try again.");
    }
  };

  const handleBackToLogin = (e) => {
    if (isModal) {
      e.preventDefault();
      switchModal('login');
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
          Forgot Password
        </h2>
        <p className="text-slate-400 text-center text-sm mb-8">We'll send you a reset link</p>

        {!isSubmitted ? (
          <form onSubmit={handleSubmit}>
            <p className="text-slate-300 mb-6 text-center text-sm">
              Enter your email address and we'll send you a link to reset your password.
            </p>

            <Input
              icon={Mail}
              type="email"
              placeholder="Email Address"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />

            {error && <p className="text-red-400 text-sm font-medium mb-4 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{error}</p>}
            {successMsg && <p className="text-green-400 text-sm font-medium mb-4 bg-green-500/10 border border-green-500/20 rounded-lg p-3">{successMsg}</p>}

            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 transition duration-200"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? <Loader className="w-6 h-6 animate-spin mx-auto" /> : "Send Reset Link"}
            </motion.button>
          </form>
        ) : (
          <div className="text-center">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500, damping: 30 }}
              className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4"
            >
              <Mail className="h-8 w-8 text-white" />
            </motion.div>
            <p className="text-slate-300 mb-6 text-sm">
              If an account exists for <span className="font-semibold text-blue-400">{email}</span>, you will receive a password reset link shortly.
            </p>
          </div>
        )}
      </div>

      <div className="px-8 py-4 bg-slate-950/50 border-t border-slate-800 flex justify-center">
        {isModal ? (
          <button
            onClick={handleBackToLogin}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors flex items-center"
          >
            <ArrowLeft className="h-4 w-4 mr-2" /> Back to Login
          </button>
        ) : (
          <Link to={"/login"} className="text-sm text-blue-400 hover:text-blue-300 transition-colors flex items-center">
            <ArrowLeft className="h-4 w-4 mr-2" /> Back to Login
          </Link>
        )}
      </div>
    </motion.div>
  );
};

export default ForgotPasswordPage;