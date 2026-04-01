import { createContext, useContext, useState } from "react";

const AuthModalContext = createContext();

export const useAuthModal = () => {
  const context = useContext(AuthModalContext);
  if (!context) {
    throw new Error("useAuthModal must be used within AuthModalProvider");
  }
  return context;
};

export const AuthModalProvider = ({ children }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalType, setModalType] = useState(null); // 'login', 'signup', 'forgot-password', 'reset-password'

  const openModal = (type) => {
    setModalType(type);
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setModalType(null);
  };

  const switchModal = (type) => {
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