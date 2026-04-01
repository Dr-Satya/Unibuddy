import { motion } from "framer-motion";
import AuthButton from "../components/AuthButton";
import { GraduationCap, Shield, Zap } from "lucide-react";

const LandingPage = () => {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="max-w-4xl w-full text-center"
      >
        {/* Logo/Icon */}
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
          className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl mb-8 shadow-2xl"
        >
          <GraduationCap className="w-10 h-10 text-white" />
        </motion.div>

        {/* Heading */}
        <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 text-transparent bg-clip-text">
          Welcome to UniBuddy
        </h1>

        <p className="text-slate-300 text-lg md:text-xl mb-12 max-w-2xl mx-auto">
          Your secure student authentication platform. Join thousands of students managing their academic journey with ease.
        </p>

        {/* Auth Buttons */}
        <div className="flex flex-wrap gap-4 justify-center mb-16">
          <AuthButton variant="login" />
          <AuthButton variant="signup" />
        </div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16"
        >
          <div className="p-6 bg-slate-900/50 backdrop-blur-xl rounded-xl border border-slate-800 hover:border-blue-500/50 transition-all">
            <div className="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <Shield className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Secure</h3>
            <p className="text-slate-400 text-sm">
              Enterprise-grade security with encrypted data and secure authentication
            </p>
          </div>

          <div className="p-6 bg-slate-900/50 backdrop-blur-xl rounded-xl border border-slate-800 hover:border-purple-500/50 transition-all">
            <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <Zap className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Fast</h3>
            <p className="text-slate-400 text-sm">
              Lightning-fast authentication with instant access to your dashboard
            </p>
          </div>

          <div className="p-6 bg-slate-900/50 backdrop-blur-xl rounded-xl border border-slate-800 hover:border-indigo-500/50 transition-all">
            <div className="w-12 h-12 bg-indigo-600/20 rounded-lg flex items-center justify-center mb-4 mx-auto">
              <GraduationCap className="w-6 h-6 text-indigo-400" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Student-Focused</h3>
            <p className="text-slate-400 text-sm">
              Built specifically for students with features tailored to academic needs
            </p>
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default LandingPage;
