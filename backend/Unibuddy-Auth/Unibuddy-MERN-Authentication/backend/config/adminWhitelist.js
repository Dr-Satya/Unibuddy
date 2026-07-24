// backend/config/adminWhitelist.js

// Add admin emails to this array
export const ADMIN_WHITELIST = [
  '230160223057.saafin@gdgu.org',
  '230160203052.kiyosha@gdgu.org',
  'samkit@gdgu.org',
  'kiyosha@gdgu.org',
  'satya.prakash@gdgu.org',
  'saafinhasan27@gdgu.org'
  // Add more admin emails as needed
];

// Function to check if email is admin
export const isAdminEmail = (email) => {
  return ADMIN_WHITELIST.includes(email.toLowerCase());
};