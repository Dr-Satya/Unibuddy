import { create } from "zustand";
import axios from "axios";

const API_URL = "http://localhost:5000/api/auth";

axios.defaults.withCredentials = true;

export const useAuthStore = create((set) => ({
  user: null,
  isAuthenticated: false,
  error: null,
  isLoading: false,
  isCheckingAuth: true,
  message: null,

  signup: async (email, password, fatherName, motherName, contactNumber, photo, collegeIdCard) => {
    set({ isLoading: true, error: null });
    try {
      const response = await axios.post(
        `${API_URL}/signup`,
        { 
          email, 
          password, 
          fatherName, 
          motherName, 
          contactNumber, 
          photo, 
          collegeIdCard 
        },
        { withCredentials: true }
      );
      set({
        user: response.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
    } catch (error) {
      console.error(error.response?.data);
      set({
        error: error.response?.data?.message || "Error signing up",
        isLoading: false,
      });
      throw error;
    }
  },

  login: async (email, password) => {
    set({ isLoading: true, error: null });
    try {
      const response = await axios.post(
        `${API_URL}/login`,
        { email, password },
        { withCredentials: true }
      );
      set({
        user: response.data.user,
        isAuthenticated: true,
        error: null,
        isLoading: false,
      });
      return response.data.user;
    } catch (error) {
      console.error(error.response?.data);
      set({
        error: error.response?.data?.message || "Error logging in",
        isLoading: false,
      });
      throw error;
    }
  },

  logout: async () => {
    set({ isLoading: true, error: null });
    try {
      await axios.post(`${API_URL}/logout`, {}, { withCredentials: true });
      set({
        user: null,
        isAuthenticated: false,
        error: null,
        isLoading: false,
      });
    } catch (error) {
      console.error(error.response?.data);
      set({ error: "Error logging out", isLoading: false });
      throw error;
    }
  },

  verifyEmail: async (code) => {
    set({ isLoading: true, error: null });
    try {
      const response = await axios.post(
        `${API_URL}/verify-email`,
        { code },
        { withCredentials: true }
      );
      set({
        user: response.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
      return response.data.user;
    } catch (error) {
      console.error(error.response?.data);
      set({
        error: error.response?.data?.message || "Error verifying email",
        isLoading: false,
      });
      throw error;
    }
  },

  checkAuth: async () => {
    set({ isCheckingAuth: true });

    try {
      const res = await axios.get(`${API_URL}/check-auth`, {
        withCredentials: true,
      });

      set({
        user: res.data.user,
        isAuthenticated: true,
        isCheckingAuth: false,
      });
    } catch (error) {
      set({
        user: null,
        isAuthenticated: false,
        isCheckingAuth: false,
      });
    }
  },

  forgotPassword: async (email) => {
    try {
      set({ isLoading: true });
      const res = await axios.post(`${API_URL}/forgot-password`, { email });
      set({ isLoading: false });
      return res.data;
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  resetPassword: async (token, password) => {
    try {
      set({ isLoading: true });
      const res = await axios.post(`${API_URL}/reset-password/${token}`, {
        password,
      });
      set({ isLoading: false });
      return res.data;
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },
}));