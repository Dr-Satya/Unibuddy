const Input = ({ icon: Icon, ...props }) => {
	return (
		<div className='relative mb-6'>
			<div className='absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none'>
				<Icon className='size-5 text-blue-400' />
			</div>
			<input
				{...props}
				className='w-full pl-10 pr-3 py-2 bg-slate-800 rounded-lg border border-slate-700 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 text-white placeholder-slate-400 transition-all duration-200'
			/>
		</div>
	);
};
export default Input;
