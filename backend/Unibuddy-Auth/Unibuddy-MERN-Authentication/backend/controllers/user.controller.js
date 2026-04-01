import { User } from '../models/user.model.js';
import bcryptjs from 'bcryptjs';

// Get all users
export const getAllUsers = async (req, res) => {
  try {
    const users = await User.find().select('-password -verificationToken -resetPasswordToken');
    res.json(users);
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};

// Create user (Admin only)
export const createUser = async (req, res) => {
  try {
    const { 
      email, 
      password, 
      fatherName, 
      motherName, 
      contactNumber, 
      photo, 
      collegeIdCard,
      role = 'STUDENT'
    } = req.body;

    // Validate GDGU email
    if (!email.endsWith('@gdgu.org')) {
      return res.status(400).json({ 
        success: false, 
        message: 'Only GDGU email addresses (@gdgu.org) are allowed' 
      });
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ 
        success: false, 
        message: 'User with this email already exists' 
      });
    }

    // Hash password
    const hashedPassword = await bcryptjs.hash(password, 10);

    // Create new user
    const user = new User({
      email,
      password: hashedPassword,
      fatherName,
      motherName,
      contactNumber,
      photo: photo || '',
      collegeIdCard: collegeIdCard || '',
      role,
      isVerified: true, // Admin-created users are auto-verified
    });

    await user.save();
    
    // Don't send password in response
    const userResponse = user.toObject();
    delete userResponse.password;
    
    res.status(201).json({ 
      success: true, 
      user: userResponse 
    });
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      message: error.message 
    });
  }
};

// Update user
export const updateUser = async (req, res) => {
  try {
    const { id } = req.params;
    const updateData = { ...req.body };
    
    // If password is provided, hash it
    if (updateData.password && updateData.password.trim() !== '') {
      updateData.password = await bcryptjs.hash(updateData.password, 10);
    } else {
      // Remove password field if empty
      delete updateData.password;
    }

    const user = await User.findByIdAndUpdate(
      id,
      updateData,
      { new: true, runValidators: true }
    ).select('-password -verificationToken -resetPasswordToken');

    if (!user) {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }

    res.json({ success: true, user });
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      message: error.message 
    });
  }
};

// Delete user
export const deleteUser = async (req, res) => {
  try {
    const { id } = req.params;
    
    const user = await User.findByIdAndDelete(id);
    
    if (!user) {
      return res.status(404).json({ 
        success: false, 
        message: 'User not found' 
      });
    }

    res.json({ 
      success: true, 
      message: 'User deleted successfully' 
    });
  } catch (error) {
    res.status(500).json({ 
      success: false, 
      message: error.message 
    });
  }
};