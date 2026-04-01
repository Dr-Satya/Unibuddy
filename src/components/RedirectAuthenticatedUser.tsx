import { Navigate } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';

interface RedirectAuthenticatedUserProps {
  children: React.ReactNode;
}

const RedirectAuthenticatedUser = ({ children }: RedirectAuthenticatedUserProps) => {
  const { isAuthenticated, user, isCheckingAuth } = useAuthStore();

  if (isCheckingAuth) return null;

  if (isAuthenticated && user?.isVerified) {
    if (user.role === 'ADMIN') {
      return <Navigate to="/admin" replace />;
    }
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
};

export default RedirectAuthenticatedUser;
