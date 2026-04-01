import { useAuthModal } from "../context/AuthModalContext";
import AuthModal from "./AuthModal";
import LoginPage from "../pages/LoginPage";
import SignUpPage from "../pages/SignUpPage";
import ForgotPasswordPage from "../pages/ForgotPasswordPage";

const AuthModalManager = () => {
  const { isModalOpen, modalType, closeModal } = useAuthModal();

  const renderModalContent = () => {
    switch (modalType) {
      case 'login':
        return <LoginPage isModal={true} />;
      case 'signup':
        return <SignUpPage isModal={true} />;
      case 'forgot-password':
        return <ForgotPasswordPage isModal={true} />;
      default:
        return null;
    }
  };

  return (
    <AuthModal isOpen={isModalOpen} onClose={closeModal}>
      {renderModalContent()}
    </AuthModal>
  );
};

export default AuthModalManager;