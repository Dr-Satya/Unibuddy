const requiredEnv = (name: string): string => {
  const value = import.meta.env[name as keyof ImportMetaEnv];
  if (!value || !String(value).trim()) {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return String(value).trim();
};

export const backendBaseUrl = requiredEnv('VITE_BACKEND_URL');
export const authBaseUrl = requiredEnv('VITE_AUTH_API_URL');

export const studentsBaseUrl = `${authBaseUrl}/students`;
export const authEndpointsBaseUrl = `${authBaseUrl}/auth`;
export const chatbotBaseUrl = backendBaseUrl;

export const chatUrl = `${chatbotBaseUrl}/chat`;
export const chatStreamUrl = `${chatbotBaseUrl}/chat/stream`;
export const mentorUploadUrl = `${backendBaseUrl}/mentor-mentee`;
export const timetableUploadUrl = `${backendBaseUrl}/timetable`;
