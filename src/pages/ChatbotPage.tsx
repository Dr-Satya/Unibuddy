import { useAuthStore } from '../store/authStore';
import { useNavigate } from 'react-router-dom';
import Chatbot from '../components/Chatbot';
import toast from 'react-hot-toast';

const ChatbotPage = () => {
  const { user, logout } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = async () => {
    try {
      await logout();
      toast.success('Logged out successfully');
      navigate('/login');
    } catch (error) {
      toast.error('Logout failed');
    }
  };

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100">
      {/* Background Accents */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute -top-32 -right-32 w-[700px] h-[700px] bg-indigo-950/30 blur-[160px] rounded-full"></div>
        <div className="absolute -bottom-40 -left-40 w-[700px] h-[700px] bg-yellow-900/10 blur-[160px] rounded-full"></div>
      </div>

      {/* Header */}
      <div className="relative z-10 bg-gray-900 bg-opacity-50 backdrop-filter backdrop-blur-xl border-b border-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-600 text-transparent bg-clip-text">
                UniBuddy Chatbot
              </h1>
              <p className="text-sm text-gray-400">Welcome, {user?.email}</p>
            </div>
            <button
              onClick={handleLogout}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
            >
              Logout
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-gray-800 bg-opacity-50 backdrop-filter backdrop-blur-xl rounded-2xl shadow-xl p-8">
          <h2 className="text-xl font-semibold mb-4">Chat with UniBuddy</h2>
          <p className="text-gray-400 mb-6">
            Ask me anything about the university, courses, faculty, fees, and more!
          </p>
          
          {/* Info Box */}
          <div className="bg-blue-900 bg-opacity-30 border border-blue-700 rounded-lg p-4 mb-6">
            <p className="text-sm text-blue-300">
              💡 The chatbot widget is available at the bottom-right corner of your screen.
              Click the icon to start chatting!
            </p>
          </div>
        </div>
      </div>

      {/* Chatbot Widget */}
      <Chatbot />
    </div>
  );
};

export default ChatbotPage;
