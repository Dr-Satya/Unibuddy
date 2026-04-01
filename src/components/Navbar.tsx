import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { useAuthModal } from '../context/AuthModalContext';
import { MessageCircle } from 'lucide-react';

const Navbar = () => {
  const navigate = useNavigate();
  const { isAuthenticated, user, logout } = useAuthStore();
  const { openModal } = useAuthModal();

  const handleDashboard = () => {
    if (user?.role === 'ADMIN') {
      navigate('/admin');
    } else {
      navigate('/'); // Student goes to home
    }
  };

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  const handleChatbotClick = () => {
    if (!isAuthenticated) {
      openModal('login');
    } else {
      // User is authenticated, chatbot widget will be available on their page
      // Optionally scroll to chatbot or show a message
      const chatbotButton = document.querySelector('[aria-label="Open chat"]') as HTMLButtonElement;
      if (chatbotButton) {
        chatbotButton.click();
      }
    }
  };

  return (
    <nav className="fixed top-0 w-full z-30 backdrop-blur-xl bg-[#05070c]/70 border-b border-white/5">
      <div className="max-w-7xl mx-auto px-10 py-6 flex justify-between items-center">
        <div className="flex items-center gap-4 cursor-pointer" onClick={() => navigate('/')}>
          <div className="w-11 h-11 rounded-full bg-[#d4af37] text-black font-black flex items-center justify-center">
            UB
          </div>
          <div>
            <h1 className="font-bold leading-none">UniBuddy</h1>
            <p className="text-[10px] tracking-[0.3em] text-[#d4af37] mt-1">
              AI COMMAND CENTER
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <div className="hidden md:flex gap-10 text-[11px] tracking-widest uppercase text-slate-500">
            <span className="hover:text-[#d4af37] cursor-pointer">Documentation</span>
            <span className="hover:text-[#d4af37] cursor-pointer">Security</span>
            <span className="hover:text-[#d4af37] cursor-pointer">Campus Map</span>
            <span className="text-emerald-400">● Network Optimal</span>
          </div>

          {/* Chatbot Icon */}
          <button
            onClick={handleChatbotClick}
            className="p-2 rounded-full bg-gradient-to-r from-blue-500 to-cyan-600 hover:from-blue-600 hover:to-cyan-700 transition-all hover:scale-110"
            title="Chat with UniBuddy"
          >
            <MessageCircle className="w-5 h-5 text-white" />
          </button>

          {isAuthenticated ? (
            <div className="flex gap-3 items-center">
              <span className="text-sm text-gray-400">{user?.email}</span>
              <button
                onClick={handleDashboard}
                className="px-4 py-2 text-sm font-medium bg-gradient-to-r from-blue-500 to-cyan-600 text-white rounded-lg hover:from-blue-600 hover:to-cyan-700 transition-all"
              >
                {user?.role === 'ADMIN' ? 'Admin Panel' : 'Home'}
              </button>
              <button
                onClick={handleLogout}
                className="px-4 py-2 text-sm font-medium text-white hover:text-red-400 transition-colors"
              >
                Logout
              </button>
            </div>
          ) : (
            <div className="flex gap-3">
              <button
                onClick={() => openModal('login')}
                className="px-4 py-2 text-sm font-medium text-white hover:text-[#d4af37] transition-colors"
              >
                Login
              </button>
              <button
                onClick={() => openModal('signup')}
                className="px-4 py-2 text-sm font-medium bg-gradient-to-r from-blue-500 to-cyan-600 text-white rounded-lg hover:from-blue-600 hover:to-cyan-700 transition-all"
              >
                Sign Up
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
