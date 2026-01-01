const Features = () => {
  return (
    <section className="max-w-7xl mx-auto px-10 py-24 grid md:grid-cols-3 gap-12 border-t border-white/5 mt-28 relative z-10">

      <div className="p-10 rounded-[2.5rem] bg-[#0e1322]/80 backdrop-blur-xl border border-white/5 hover:border-[#d4af37]/30 transition">
        <div className="text-4xl mb-6">🎓</div>
        <h3 className="text-xl font-bold mb-4 text-[#d4af37]">Student Hub</h3>
        <p className="text-slate-500 text-sm leading-relaxed">
          Attendance, grades, scholarships & academic guidance.
        </p>
      </div>

      <div className="p-10 rounded-[2.5rem] bg-[#0e1322]/80 backdrop-blur-xl border border-white/5 hover:border-[#d4af37]/30 transition">
        <div className="text-4xl mb-6">🧑‍🏫</div>
        <h3 className="text-xl font-bold mb-4 text-[#d4af37]">Faculty Tools</h3>
        <p className="text-slate-500 text-sm leading-relaxed">
          Classroom allocation, schedules & academic workflows.
        </p>
      </div>

      <div className="p-10 rounded-[2.5rem] bg-[#0e1322]/80 backdrop-blur-xl border border-white/5 hover:border-[#d4af37]/30 transition">
        <div className="text-4xl mb-6">🛡️</div>
        <h3 className="text-xl font-bold mb-4 text-[#d4af37]">Admin Control</h3>
        <p className="text-slate-500 text-sm leading-relaxed">
          Governance dashboards & strategic insights.
        </p>
      </div>

    </section>
  );
};

export default Features;
