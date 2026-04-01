import bcryptjs from "bcryptjs";
import crypto from "crypto";

import { generateTokenAndSetCookie } from "../utils/generateTokenAndSetCookie.js";
import {
  sendPasswordResetEmail,
  sendResetSuccessEmail,
  sendVerificationEmail,
  sendWelcomeEmail,
} from "../mailtrap/emails.js";
import { User } from "../models/user.model.js";
import { Student } from "../models/student.model.js";
import { isAdminEmail } from "../config/adminWhitelist.js";

export const signup = async (req, res) => {
  const { email, password, fatherName, motherName, contactNumber, photo, collegeIdCard } = req.body;

  try {
    // Validate all required fields
    if (!email || !password || !fatherName || !motherName || !contactNumber || !photo || !collegeIdCard) {
      throw new Error("All fields are required");
    }

    // Validate GDGU email domain
    if (!email.endsWith('@gdgu.org')) {
      return res.status(400).json({ 
        success: false, 
        message: "Only GDGU email addresses (@gdgu.org) are allowed" 
      });
    }

    // Check if user already exists
    const userAlreadyExists = await User.findOne({ email });
    if (userAlreadyExists) {
      return res.status(400).json({ 
        success: false, 
        message: "User already exists" 
      });
    }

    // Also check and clean orphaned student record
    const orphanedStudent = await Student.findOne({ email });
    if (orphanedStudent) {
      await Student.deleteOne({ email });
    }

    // Hash password
    const hashedPassword = await bcryptjs.hash(password, 10);
    
    // Generate OTP (6-digit)
    const verificationToken = Math.floor(100000 + Math.random() * 900000).toString();

    // Determine role based on admin whitelist
    const role = isAdminEmail(email) ? 'ADMIN' : 'STUDENT';

    // Create new user
    const user = new User({
      email,
      password: hashedPassword,
      fatherName,
      motherName,
      contactNumber,
      photo,
      collegeIdCard,
      role,
      verificationToken,
      verificationTokenExpiresAt: Date.now() + 15 * 60 * 1000, // 15 minutes
    });

    await user.save();

    // If user is a STUDENT, also create a Student record for Admin Panel
    if (role === 'STUDENT') {
      try {
        // Check if student record already exists (cleanup any orphaned records)
        const existingStudent = await Student.findOne({ email });
        if (!existingStudent) {
          // Extract name from email (before @)
          const name = email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
          
          // Generate a temporary roll number (can be updated by admin later)
          const rollNo = `TEMP-${Date.now().toString().slice(-6)}`;
          
          const student = new Student({
            name: name,
            email: email,
            rollNo: rollNo,
            department: "Not Assigned",
            year: "1",
            fatherName: fatherName,
            motherName: motherName,
            contactNumber: contactNumber,
            photo: photo,
            collegeIdCard: collegeIdCard,
            status: 'ACTIVE',
          });

          await student.save();
        }
      } catch (studentErr) {
        // If student creation fails, still continue - don't block signup
        console.error("Student record creation failed (non-critical):", studentErr.message);
      }
    }

    // Generate JWT token
    generateTokenAndSetCookie(res, user._id);

    // Send OTP email
    await sendVerificationEmail(user.email, verificationToken);

    res.status(201).json({
      success: true,
      message: "User created successfully. Please verify your email with the OTP sent.",
      user: {
        ...user._doc,
        password: undefined,
      },
    });
  } catch (error) {
    res.status(400).json({ success: false, message: error.message });
  }
};

export const verifyEmail = async (req, res) => {
  const { code } = req.body;
  try {
    const user = await User.findOne({
      verificationToken: code,
      verificationTokenExpiresAt: { $gt: Date.now() },
    });

    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "Invalid or expired verification code" 
      });
    }

    user.isVerified = true;
    user.verificationToken = undefined;
    user.verificationTokenExpiresAt = undefined;
    await user.save();

    await sendWelcomeEmail(user.email, user.email.split('@')[0]);

    res.status(200).json({
      success: true,
      message: "Email verified successfully",
      user: {
        ...user._doc,
        password: undefined,
      },
    });
  } catch (error) {
    console.log("error in verifyEmail ", error);
    res.status(500).json({ success: false, message: "Server error" });
  }
};

export const login = async (req, res) => {
  const { email, password } = req.body;
  try {
    // Validate GDGU email
    if (!email.endsWith('@gdgu.org')) {
      return res.status(400).json({ 
        success: false, 
        message: "Only GDGU email addresses are allowed" 
      });
    }

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "Invalid credentials" 
      });
    }

    // Check if email is verified
    if (!user.isVerified) {
      return res.status(400).json({ 
        success: false, 
        message: "Please verify your email before logging in" 
      });
    }

    const isPasswordValid = await bcryptjs.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({ 
        success: false, 
        message: "Invalid credentials" 
      });
    }

    generateTokenAndSetCookie(res, user._id);

    user.lastLogin = new Date();
    await user.save();

    res.status(200).json({
      success: true,
      message: "Logged in successfully",
      user: {
        ...user._doc,
        password: undefined,
      },
    });
  } catch (error) {
    console.log("Error in login ", error);
    res.status(400).json({ success: false, message: error.message });
  }
};

export const logout = async (req, res) => {
  res.clearCookie("token");
  res.status(200).json({ success: true, message: "Logged out successfully" });
};

export const forgotPassword = async (req, res) => {
  const { email } = req.body;
  try {
    const user = await User.findOne({ email });

    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "User not found" 
      });
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(20).toString("hex");
    const resetTokenExpiresAt = Date.now() + 1 * 60 * 60 * 1000; // 1 hour

    user.resetPasswordToken = resetToken;
    user.resetPasswordExpiresAt = resetTokenExpiresAt;

    await user.save();

    // send email
    await sendPasswordResetEmail(
      user.email, 
      `${process.env.CLIENT_URL}/reset-password/${resetToken}`
    );

    res.status(200).json({ 
      success: true, 
      message: "Password reset link sent to your email" 
    });
  } catch (error) {
    console.log("Error in forgotPassword ", error);
    res.status(400).json({ success: false, message: error.message });
  }
};

export const resetPassword = async (req, res) => {
  try {
    const { token } = req.params;
    const { password } = req.body;

    const user = await User.findOne({
      resetPasswordToken: token,
      resetPasswordExpiresAt: { $gt: Date.now() },
    });

    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "Invalid or expired reset token" 
      });
    }

    // update password
    const hashedPassword = await bcryptjs.hash(password, 10);

    user.password = hashedPassword;
    user.resetPasswordToken = undefined;
    user.resetPasswordExpiresAt = undefined;
    await user.save();

    await sendResetSuccessEmail(user.email);

    res.status(200).json({ 
      success: true, 
      message: "Password reset successful" 
    });
  } catch (error) {
    console.log("Error in resetPassword ", error);
    res.status(400).json({ success: false, message: error.message });
  }
};

export const checkAuth = async (req, res) => {
  try {
    const user = await User.findById(req.userId).select("-password");
    if (!user) {
      return res.status(400).json({ 
        success: false, 
        message: "User not found" 
      });
    }

    res.status(200).json({ success: true, user });
  } catch (error) {
    console.log("Error in checkAuth ", error);
    res.status(400).json({ success: false, message: error.message });
  }
};