import Navbar from '../components/Navbar';
import Home from '../components/Home';
import Features from '../components/Features';
import Footer from '../components/Footer';
import Chatbot from '../components/Chatbot';

const HomePage = () => {
  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100 overflow-x-hidden">
      {/* Background Accents */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute -top-32 -right-32 w-[700px] h-[700px] bg-indigo-950/30 blur-[160px] rounded-full"></div>
        <div className="absolute -bottom-40 -left-40 w-[700px] h-[700px] bg-yellow-900/10 blur-[160px] rounded-full"></div>
      </div>

      <Navbar />
      <Home />
      <Features />
      <Footer />
      
      {/* Chatbot Widget - Shows on landing page */}
      <Chatbot />
    </div>
  );
};

export default HomePage;
