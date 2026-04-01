import { motion } from "framer-motion";
import { LogIn, UserPlus } from "lucide-react";
import { useAuthModal } from "../context/AuthModalContext";

const AuthButton = ({ variant = "login", className = "" }) => {
  const { openModal } = useAuthModal();

  const handleClick = () => {
    openModal(variant);
  };

  const isLogin = variant === "login";

  return (
    <motion.button
      onClick={handleClick}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className={`
        flex items-center gap-2 px-6 py-3 rounded-lg font-semibold
        transition-all duration-200 shadow-lg
        ${isLogin 
          ? "bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white" 
          : "bg-slate-800 hover:bg-slate-700 text-white border border-slate-700"
        }
        ${className}
      `}
    >
      {isLogin ? <LogIn className="w-5 h-5" /> : <UserPlus className="w-5 h-5" />}
      {isLogin ? "Sign In" : "Sign Up"}
    </motion.button>
  );
};

export default AuthButton;
