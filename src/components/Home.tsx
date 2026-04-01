import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { useAuthModal } from '../context/AuthModalContext';

const Home = () => {
  const navigate = useNavigate();
  const { isAuthenticated, user } = useAuthStore();
  const { openModal } = useAuthModal();

  const handleStartConversation = () => {
    if (isAuthenticated) {
      // Redirect based on role
      if (user?.role === 'ADMIN') {
        navigate('/admin');
      } else {
        // Student stays on home page, just scroll to top or show message
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } else {
      openModal('login');
    }
  };

  return (
    <main className="relative z-10 pt-40">
      <section className="max-w-7xl mx-auto px-10 text-center">

        <div className="inline-flex px-6 py-2 rounded-full bg-[#d4af37]/10 border border-[#d4af37]/30
                        text-[#d4af37] text-[10px] font-black mb-12 tracking-[0.4em] uppercase">
          GD Goenka University Gurgaon
        </div>

        <h1 className="text-6xl md:text-9xl font-black mb-10 leading-[0.85]">
          INTELLIGENT <br />
          <span className="text-[#d4af37] drop-shadow-[0_0_18px_rgba(212,175,55,0.25)]">
            UNIBUDDY.
          </span>
        </h1>

        <p className="text-xl md:text-2xl text-slate-400 mb-16 max-w-3xl mx-auto">
          One portal, infinite intelligence. Access admissions, academic records,
          and facility management through a unified AI-driven interface.
        </p>

        <button
          onClick={handleStartConversation}
          className="px-12 py-5 bg-[#d4af37] text-[#0f172a] rounded-2xl font-black text-xl
                     shadow-[0_20px_50px_-10px_rgba(212,175,55,0.4)]
                     hover:scale-105 transition uppercase tracking-widest">
          {isAuthenticated ? (user?.role === 'ADMIN' ? 'Open Admin Panel' : 'Start Chatting') : 'Get Started'}
        </button>

      </section>
    </main>
  );
};

export default Home;
