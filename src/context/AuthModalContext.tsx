import { createContext, useContext, useState, type ReactNode } from "react";

interface AuthModalContextType {
  isModalOpen: boolean;
  modalType: string | null;
  openModal: (type: string) => void;
  closeModal: () => void;
  switchModal: (type: string) => void;
}

const AuthModalContext = createContext<AuthModalContextType | undefined>(undefined);

export const useAuthModal = () => {
  const context = useContext(AuthModalContext);
  if (!context) {
    throw new Error("useAuthModal must be used within AuthModalProvider");
  }
  return context;
};

interface AuthModalProviderProps {
  children: ReactNode;
}

export const AuthModalProvider = ({ children }: AuthModalProviderProps) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalType, setModalType] = useState<string | null>(null); // 'login', 'signup', 'forgot-password', 'reset-password'

  const openModal = (type: string) => {
    setModalType(type);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setModalType(null);
  };

  const switchModal = (type: string) => {
    setModalType(type);
  };

  return (
    <AuthModalContext.Provider
      value={{
        isModalOpen,
        modalType,
        openModal,
        closeModal,
        switchModal,
      }}
    >
      {children}
    </AuthModalContext.Provider>
  );
};
