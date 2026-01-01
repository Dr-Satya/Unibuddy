const Navbar = () => {
  return (
    <nav className="fixed top-0 w-full z-30 backdrop-blur-xl bg-[#05070c]/70 border-b border-white/5">
      <div className="max-w-7xl mx-auto px-10 py-6 flex justify-between items-center">
        <div className="flex items-center gap-4">
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

        <div className="hidden md:flex gap-10 text-[11px] tracking-widest uppercase text-slate-500">
          <span className="hover:text-[#d4af37] cursor-pointer">Documentation</span>
          <span className="hover:text-[#d4af37] cursor-pointer">Security</span>
          <span className="hover:text-[#d4af37] cursor-pointer">Campus Map</span>
          <span className="text-emerald-400">● Network Optimal</span>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
