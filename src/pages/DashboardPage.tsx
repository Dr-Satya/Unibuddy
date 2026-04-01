import { motion } from "framer-motion";
import { useAuthStore } from "../store/authStore";
import { formatDate } from "../utils/date";
import Confetti from "../particle/Confetti";
import { useEffect, useState } from "react";
import Chatbot from "../components/Chatbot";
import Navbar from "../components/Navbar";

const DashboardPage = () => {
    const { user, logout } = useAuthStore();
    const [showConfetti, setShowConfetti] = useState(true);

    useEffect(() => {
        const timer = setTimeout(() => {
            setShowConfetti(false);
        }, 3000);

        return () => clearTimeout(timer);
    }, [user]);

    const handleLogout = () => {
        logout();
    };

    return (
        <div className="min-h-screen bg-[#080c14] text-slate-100">
            {/* Background Accents */}
            <div className="fixed inset-0 pointer-events-none z-0">
                <div className="absolute -top-32 -right-32 w-[700px] h-[700px] bg-indigo-950/30 blur-[160px] rounded-full"></div>
                <div className="absolute -bottom-40 -left-40 w-[700px] h-[700px] bg-yellow-900/10 blur-[160px] rounded-full"></div>
            </div>

            <Navbar />
            
            {showConfetti && <Confetti />}
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.5 }}
                className="max-w-md w-full mx-auto mt-10 p-8 bg-slate-900/95 backdrop-filter backdrop-blur-xl rounded-xl shadow-2xl border border-slate-800 z-50 relative"
            >
                <h2 className="text-3xl font-bold mb-6 text-center bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 text-transparent bg-clip-text">
                    Dashboard
                </h2>

                <div className="space-y-6">
                    {/* Profile Information */}
                    <motion.div
                        className="p-4 bg-slate-800/50 rounded-lg border border-slate-700"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2 }}
                    >
                        <h3 className="text-xl font-semibold text-blue-400 mb-3">Profile Information</h3>
                        <p className="text-slate-300">Name: {user?.name || user?.email?.split('@')[0] || 'User'}</p>
                        <p className="text-slate-300">Email: {user?.email}</p>
                    </motion.div>

                    {/* Account Activity */}
                    <motion.div
                        className="p-4 bg-slate-800/50 rounded-lg border border-slate-700"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                    >
                        <h3 className="text-xl font-semibold text-blue-400 mb-3">Account Activity</h3>
                        <p className="text-slate-300">
                            <span className="font-bold">Joined: </span>
                            {user?.createdAt ? new Date(user.createdAt).toLocaleDateString("en-US", {
                                year: "numeric",
                                month: "long",
                                day: "numeric",
                            }) : 'N/A'}
                        </p>
                        <p className="text-slate-300">
                            <span className="font-bold">Last Login: </span>
                            {user?.lastLogin ? formatDate(user.lastLogin) : 'N/A'}
                        </p>
                    </motion.div>
                </div>

                {/* Logout Button */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    className="mt-4"
                >
                    <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        onClick={handleLogout}
                        className="w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white 
                        font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700
                        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                    >
                        Logout
                    </motion.button>
                </motion.div>
            </motion.div>

            {/* Chatbot Widget */}
            <Chatbot />
        </div>
    );
};

export default DashboardPage;
