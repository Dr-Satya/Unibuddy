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
    // Timetable fields
    degree: {
      type: String,
      default: '',   // e.g. BTECH, BCA, MCA
    },
    branch: {
      type: String,
      default: '',   // e.g. CSE, ECE, ME
    },
    year: {
      type: String,
      default: '',   // 1 / 2 / 3 / 4
    },
    section: {
      type: String,
      default: '',   // e.g. BTECH|2|CSE_B2
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