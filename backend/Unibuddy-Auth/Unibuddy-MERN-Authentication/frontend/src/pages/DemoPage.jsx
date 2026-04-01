import { motion } from "framer-motion";
import AuthButton from "../components/AuthButton";
import { Sparkles, Zap, Shield } from "lucide-react";

/**
 * Demo page showcasing the new authentication UI
 * This page demonstrates how easy it is to integrate auth buttons
 */

const DemoPage = () => {
  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-4xl w-full"
      >
        {/* Header */}
        <div className="text-center mb-12">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: "spring", stiffness: 200 }}
            className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl mb-6 shadow-2xl"
          >
            <Sparkles className="w-8 h-8 text-white" />
          </motion.div>
          
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 text-transparent bg-clip-text">
            New Professional UI
          </h1>
          <p className="text-slate-400 text-lg">
            Click any button below to see the modal authentication in action!
          </p>
        </div>

        {/* Main Demo Section */}
        <div className="grid md:grid-cols-2 gap-6 mb-12">
          {/* Login Demo */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="p-8 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 hover:border-blue-500/50 transition-all"
          >
            <div className="w-12 h-12 bg-blue-600/20 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-blue-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">Sign In</h3>
            <p className="text-slate-400 mb-6">
              Existing users can sign in with their credentials. The modal opens instantly with smooth animations.
            </p>
            <AuthButton variant="login" className="w-full" />
          </motion.div>

          {/* Signup Demo */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="p-8 bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 hover:border-purple-500/50 transition-all"
          >
            <div className="w-12 h-12 bg-purple-600/20 rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-purple-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-2">Sign Up</h3>
            <p className="text-slate-400 mb-6">
              New users can create an account. All forms are now modal-based for a better user experience.
            </p>
            <AuthButton variant="signup" className="w-full" />
          </motion.div>
        </div>

        {/* Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-slate-900/50 backdrop-blur-xl rounded-2xl border border-slate-800 p-8"
        >
          <h2 className="text-2xl font-bold text-white mb-6 text-center">
            What's New?
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="text-4xl mb-3">🎨</div>
              <h4 className="font-semibold text-white mb-2">Dark Theme</h4>
              <p className="text-sm text-slate-400">
                Professional slate gray with blue/purple accents
              </p>
            </div>

            <div className="text-center">
              <div className="text-4xl mb-3">✨</div>
              <h4 className="font-semibold text-white mb-2">Modal Based</h4>
              <p className="text-sm text-slate-400">
                No more full-page redirects, everything in modals
              </p>
            </div>

            <div className="text-center">
              <div className="text-4xl mb-3">🚀</div>
              <h4 className="font-semibold text-white mb-2">One Click</h4>
              <p className="text-sm text-slate-400">
                Simple AuthButton component, use anywhere
              </p>
            </div>
          </div>
        </motion.div>

        {/* Code Example */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 p-6 bg-slate-950/50 backdrop-blur-xl rounded-xl border border-slate-800"
        >
          <h3 className="text-lg font-semibold text-white mb-3">
            💻 Usage Example
          </h3>
          <pre className="text-sm text-slate-300 overflow-x-auto">
{`import AuthButton from "./components/AuthButton";

// That's it! Just use the button anywhere
<AuthButton variant="login" />
<AuthButton variant="signup" />`}
          </pre>
        </motion.div>

        {/* Try Both Buttons */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-8 text-center"
        >
          <p className="text-slate-400 mb-4">Try them together:</p>
          <div className="flex gap-4 justify-center">
            <AuthButton variant="login" />
            <AuthButton variant="signup" />
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default DemoPage;
