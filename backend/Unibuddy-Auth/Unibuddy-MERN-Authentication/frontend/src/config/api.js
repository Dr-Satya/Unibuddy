const requiredEnv = (name) => {
  const value = import.meta.env[name];
  if (!value || !String(value).trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return String(value).trim();
};

export const authApiBaseUrl = requiredEnv("VITE_AUTH_API_URL");
export const chatbotBaseUrl = requiredEnv("VITE_BACKEND_URL");

export const studentsApiBaseUrl = `${authApiBaseUrl}/students`;
export const authEndpointsBaseUrl = `${authApiBaseUrl}/auth`;
export const chatUrl = `${chatbotBaseUrl}/chat`;
export const chatStreamUrl = `${chatbotBaseUrl}/chat/stream`;
export const sectionsUrl = `${chatbotBaseUrl}/sections`;
