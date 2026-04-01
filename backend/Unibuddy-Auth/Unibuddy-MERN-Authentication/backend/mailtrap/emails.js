import {
  PASSWORD_RESET_REQUEST_TEMPLATE,
  PASSWORD_RESET_SUCCESS_TEMPLATE,
  VERIFICATION_EMAIL_TEMPLATE,
} from "./emailTemplates.js";
import { transporter, sender } from "./mailer.config.js"; // updated import

// Send verification email
export const sendVerificationEmail = async (email, verificationToken) => {
  try {
    const info = await transporter.sendMail({
      from: `"${sender.name}" <${sender.email}>`,
      to: email,
      subject: "Verify your email",
      html: VERIFICATION_EMAIL_TEMPLATE.replace("{verificationCode}", verificationToken),
    });
    console.log("Verification email sent:", info.messageId);
  } catch (error) {
    console.error("Error sending verification email:", error);
    throw new Error(`Error sending verification email: ${error.message}`);
  }
};

// Send welcome email
export const sendWelcomeEmail = async (email, name) => {
  try {
    const info = await transporter.sendMail({
      from: `"${sender.name}" <${sender.email}>`,
      to: email,
      subject: "Welcome to our platform!",
      html: `Hello ${name},<br><br>Welcome to our platform. We are glad to have you onboard!`,
    });
    console.log("Welcome email sent:", info.messageId);
  } catch (error) {
    console.error("Error sending welcome email:", error);
    throw new Error(`Error sending welcome email: ${error.message}`);
  }
};

// Send password reset email
export const sendPasswordResetEmail = async (email, resetURL) => {
  try {
    const info = await transporter.sendMail({
      from: `"${sender.name}" <${sender.email}>`,
      to: email,
      subject: "Reset your password",
      html: PASSWORD_RESET_REQUEST_TEMPLATE.replace("{resetURL}", resetURL),
    });
    console.log("Password reset email sent:", info.messageId);
  } catch (error) {
    console.error("Error sending password reset email:", error);
    throw new Error(`Error sending password reset email: ${error.message}`);
  }
};

// Send password reset success email
export const sendResetSuccessEmail = async (email) => {
  try {
    const info = await transporter.sendMail({
      from: `"${sender.name}" <${sender.email}>`,
      to: email,
      subject: "Password Reset Successful",
      html: PASSWORD_RESET_SUCCESS_TEMPLATE,
    });
    console.log("Password reset success email sent:", info.messageId);
  } catch (error) {
    console.error("Error sending password reset success email:", error);
    throw new Error(`Error sending password reset success email: ${error.message}`);
  }
};
