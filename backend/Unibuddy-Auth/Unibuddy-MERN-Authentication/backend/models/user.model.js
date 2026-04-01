import mongoose from "mongoose";

const userSchema = new mongoose.Schema(
  {
    email: {
      type: String,
      required: true,
      unique: true,
      validate: {
        validator: function(v) {
          return v.endsWith('@gdgu.org');
        },
        message: 'Email must be a valid GDGU email (@gdgu.org)'
      }
    },
    password: {
      type: String,
      required: true,
    },
    // New mandatory fields
    fatherName: {
      type: String,
      required: true,
    },
    motherName: {
      type: String,
      required: true,
    },
    contactNumber: {
      type: String,
      required: true,
      validate: {
        validator: function(v) {
          return /^[0-9]{10}$/.test(v);
        },
        message: 'Contact number must be 10 digits'
      }
    },
    photo: {
      type: String, // URL or base64 string
      required: true,
    },
    collegeIdCard: {
      type: String, // URL or base64 string
      required: true,
    },
    // Role field
    role: {
      type: String,
      enum: ['STUDENT', 'ADMIN'],
      default: 'STUDENT'
    },
    lastLogin: {
      type: Date,
      default: Date.now,
    },
    isVerified: {
      type: Boolean,
      default: false,
    },
    resetPasswordToken: String,
    resetPasswordExpiresAt: Date,
    verificationToken: String,
    verificationTokenExpiresAt: Date,
  },
  { timestamps: true }
);

export const User = mongoose.model("User", userSchema);