import { motion } from "framer-motion";
import Input from "../components/Input";
import { Loader, Lock, Mail, User, Phone, Upload, X } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import PasswordStrengthMeter from "../components/PasswordStrengthMeter";
import { useAuthStore } from "../store/authStore";
import { useAuthModal } from "../context/AuthModalContext";

const SignUpPage = ({ isModal = false }) => {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    fatherName: "",
    motherName: "",
    contactNumber: "",
  });
  const [photo, setPhoto] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [collegeIdCard, setCollegeIdCard] = useState(null);
  const [idCardPreview, setIdCardPreview] = useState(null);
  
  const navigate = useNavigate();
  const { signup, error, isLoading } = useAuthStore();
  
  // Always call the hook, but only use it when isModal is true
  const modalContext = useAuthModal();
  const closeModal = isModal ? modalContext.closeModal : () => {};
  const switchModal = isModal ? modalContext.switchModal : () => {};

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handlePhotoChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setPhoto(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPhotoPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleIdCardChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setCollegeIdCard(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setIdCardPreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSignUp = async (e) => {
    e.preventDefault();

    // Validate GDGU email
    if (!formData.email.endsWith('@gdgu.org')) {
      alert('Please use your GDGU email address (@gdgu.org)');
      return;
    }

    // Validate all fields
    if (!formData.email || !formData.password || !formData.fatherName || 
        !formData.motherName || !formData.contactNumber || !photo || !collegeIdCard) {
      alert('All fields are required');
      return;
    }

    // Validate contact number
    if (!/^[0-9]{10}$/.test(formData.contactNumber)) {
      alert('Contact number must be 10 digits');
      return;
    }

    try {
      await signup(
        formData.email,
        formData.password,
        formData.fatherName,
        formData.motherName,
        formData.contactNumber,
        photoPreview, // Base64 encoded
        idCardPreview  // Base64 encoded
      );
      if (isModal) {
        closeModal(); // Close modal and navigate from parent
      }
      navigate("/verify-email");
    } catch (error) {
      console.log(error);
    }
  };

  const handleLoginClick = (e) => {
    if (isModal) {
      e.preventDefault();
      switchModal('login');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className='max-w-2xl w-full bg-slate-900/95 backdrop-filter backdrop-blur-xl rounded-2xl shadow-2xl border border-slate-800 overflow-hidden relative'
    >
      {/* Close Button - Only show in modal mode */}
      {isModal && (
        <button
          onClick={closeModal}
          className="absolute top-4 right-4 z-10 bg-slate-800/90 hover:bg-slate-700 text-slate-400 hover:text-white rounded-full p-2 transition-all duration-200 hover:scale-110"
          aria-label="Close"
        >
          <X className="w-5 h-5" />
        </button>
      )}

      <div className='p-8'>
        <h2 className='text-3xl font-bold mb-2 text-center bg-gradient-to-r from-blue-400 via-purple-400 to-indigo-400 text-transparent bg-clip-text'>
          GDGU Student Registration
        </h2>
        <p className="text-slate-400 text-center text-sm mb-8">Create your account to get started</p>

        <form onSubmit={handleSignUp} className='space-y-4'>
          <Input
            icon={Mail}
            type='email'
            name='email'
            placeholder='GDGU Email (@gdgu.org)'
            value={formData.email}
            onChange={handleInputChange}
          />
          
          <Input
            icon={Lock}
            type='password'
            name='password'
            placeholder='Password'
            value={formData.password}
            onChange={handleInputChange}
          />
          <PasswordStrengthMeter password={formData.password} />

          <Input
            icon={User}
            type='text'
            name='fatherName'
            placeholder="Father's Name"
            value={formData.fatherName}
            onChange={handleInputChange}
          />

          <Input
            icon={User}
            type='text'
            name='motherName'
            placeholder="Mother's Name"
            value={formData.motherName}
            onChange={handleInputChange}
          />

          <Input
            icon={Phone}
            type='tel'
            name='contactNumber'
            placeholder='Contact Number (10 digits)'
            value={formData.contactNumber}
            onChange={handleInputChange}
            maxLength={10}
          />

          {/* Photo Upload */}
          <div className='space-y-2'>
            <label className='flex items-center gap-2 text-slate-300 text-sm font-medium'>
              <Upload className='w-4 h-4' />
              Upload Photo
            </label>
            <input
              type='file'
              accept='image/*'
              onChange={handlePhotoChange}
              className='w-full px-3 py-2 bg-slate-800 text-white rounded-lg border border-slate-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all'
            />
            {photoPreview && (
              <img src={photoPreview} alt='Preview' className='w-32 h-32 object-cover rounded-lg mt-2 border-2 border-slate-700' />
            )}
          </div>

          {/* College ID Card Upload */}
          <div className='space-y-2'>
            <label className='flex items-center gap-2 text-slate-300 text-sm font-medium'>
              <Upload className='w-4 h-4' />
              Upload College ID Card
            </label>
            <input
              type='file'
              accept='image/*'
              onChange={handleIdCardChange}
              className='w-full px-3 py-2 bg-slate-800 text-white rounded-lg border border-slate-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20 transition-all'
            />
            {idCardPreview && (
              <img src={idCardPreview} alt='ID Preview' className='w-full h-48 object-cover rounded-lg mt-2 border-2 border-slate-700' />
            )}
          </div>

          {error && <p className='text-red-400 text-sm font-medium bg-red-500/10 border border-red-500/20 rounded-lg p-3'>{error}</p>}

          <motion.button
            className='mt-5 w-full py-3 px-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white 
            font-semibold rounded-lg shadow-lg hover:from-blue-700 hover:to-purple-700 
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
            focus:ring-offset-slate-900 transition duration-200'
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            type='submit'
            disabled={isLoading}
          >
            {isLoading ? <Loader className='animate-spin mx-auto' size={24} /> : "Create Account"}
          </motion.button>
        </form>
      </div>
      <div className='px-8 py-4 bg-slate-950/50 border-t border-slate-800 flex justify-center'>
        <p className='text-sm text-slate-400'>
          Already have an account?{" "}
          {isModal ? (
            <button
              onClick={handleLoginClick}
              className='text-blue-400 hover:text-blue-300 font-medium transition-colors'
            >
              Login
            </button>
          ) : (
            <Link to={"/login"} className='text-blue-400 hover:text-blue-300 font-medium transition-colors'>
              Login
            </Link>
          )}
        </p>
      </div>
    </motion.div>
  );
};

export default SignUpPage;