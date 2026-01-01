const Footer = () => {
  return (
    <footer className="relative z-20 py-12 border-t border-white/5 bg-black/20 backdrop-blur-md px-10
                       flex flex-col md:flex-row justify-between items-center gap-6
                       text-[10px] font-black uppercase tracking-[0.4em] text-slate-600">
      <div>© 2026 GD Goenka University Integrated AI Systems</div>
      <div className="flex gap-8">
        <span className="hover:text-white cursor-pointer transition">Privacy Protocol</span>
        <span className="hover:text-white cursor-pointer transition">Ethics Guidelines</span>
      </div>
    </footer>
  );
};

export default Footer;
