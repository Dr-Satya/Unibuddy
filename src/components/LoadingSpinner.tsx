const LoadingSpinner = () => {
  return (
    <div className="min-h-screen bg-[#080c14] flex items-center justify-center">
      <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-blue-500"></div>
    </div>
  );
};

export default LoadingSpinner;
