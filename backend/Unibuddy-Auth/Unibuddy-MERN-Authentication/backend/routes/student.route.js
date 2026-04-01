import express from "express";
import { Student } from "../models/student.model.js";
import { verifyToken } from "../middleware/verifyToken.js";
import bcrypt from "bcryptjs";

const router = express.Router();

// Middleware to check if user is admin
const isAdmin = (req, res, next) => {
  if (req.user && req.user.role === 'ADMIN') {
    next();
  } else {
    res.status(403).json({ 
      success: false, 
      message: 'Access denied. Admin only.' 
    });
  }
};

// Get all students (admin only)
router.get("/", verifyToken, isAdmin, async (req, res) => {
  try {
    const students = await Student.find().sort({ createdAt: -1 });
    res.json(students);
  } catch (error) {
    console.error("Error fetching students:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Get single student by ID
router.get("/:id", verifyToken, isAdmin, async (req, res) => {
  try {
    const student = await Student.findById(req.params.id);
    
    if (!student) {
      return res.status(404).json({ message: "Student not found" });
    }

    res.json(student);
  } catch (error) {
    console.error("Error fetching student:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Add new student (admin only)
router.post("/", verifyToken, isAdmin, async (req, res) => {
  try {
    console.log("Received student data:", req.body); // Debug log

    const {
      name,
      email,
      password,
      rollNo,
      department,
      year,
      fatherName,
      motherName,
      contactNumber,
      address,
      dateOfBirth,
      gender,
      photo,
      collegeIdCard,
      status,
    } = req.body;

    // Validation
    if (!name || !email || !rollNo || !department || !year || !fatherName || !motherName || !contactNumber) {
      console.log("Validation failed - missing fields");
      return res.status(400).json({ message: "Please provide all required fields" });
    }

    // Validate GDGU email
    if (!email.endsWith('@gdgu.org')) {
      console.log("Validation failed - invalid email");
      return res.status(400).json({ 
        message: 'Only GDGU email addresses (@gdgu.org) are allowed' 
      });
    }

    // Check if student already exists
    const existingStudent = await Student.findOne({
      $or: [{ email }, { rollNo }],
    });

    if (existingStudent) {
      console.log("Student already exists:", existingStudent.email || existingStudent.rollNo);
      if (existingStudent.email === email) {
        return res.status(400).json({ message: "Email already exists" });
      }
      if (existingStudent.rollNo === rollNo) {
        return res.status(400).json({ message: "Roll number already exists" });
      }
    }

    // Create new student
    const newStudent = new Student({
      name,
      email,
      rollNo,
      department,
      year,
      fatherName,
      motherName,
      contactNumber,
      address: address || "",
      dateOfBirth: dateOfBirth && dateOfBirth.trim() !== "" ? dateOfBirth : undefined,
      gender: gender || "",
      photo: photo || "",
      collegeIdCard: collegeIdCard || "",
      status: status || 'ACTIVE',
    });

    console.log("Attempting to save student:", newStudent);
    await newStudent.save();
    console.log("Student saved successfully:", newStudent._id);

    res.status(201).json({
      message: "Student added successfully",
      student: newStudent,
    });
  } catch (error) {
    console.error("Error adding student:", error);
    console.error("Error stack:", error.stack);
    res.status(500).json({ 
      message: "Server error", 
      error: error.message,
      details: error.errors ? Object.keys(error.errors).map(key => ({
        field: key,
        message: error.errors[key].message
      })) : null
    });
  }
});
// Update student (admin only)
router.put("/:id", verifyToken, isAdmin, async (req, res) => {
  try {
    const {
      name,
      email,
      rollNo,
      department,
      year,
      fatherName,
      motherName,
      contactNumber,
      address,
      dateOfBirth,
      gender,
      photo,
      collegeIdCard,
      status,
    } = req.body;

    const student = await Student.findById(req.params.id);
    if (!student) {
      return res.status(404).json({ message: "Student not found" });
    }

    // Check if email already exists (excluding current student)
    if (email && email !== student.email) {
      if (!email.endsWith('@gdgu.org')) {
        return res.status(400).json({ 
          message: 'Only GDGU email addresses (@gdgu.org) are allowed' 
        });
      }
      const emailExists = await Student.findOne({ email });
      if (emailExists) {
        return res.status(400).json({ message: "Email already exists" });
      }
    }

    // Check if rollNo already exists (excluding current student)
    if (rollNo && rollNo !== student.rollNo) {
      const rollNoExists = await Student.findOne({ rollNo });
      if (rollNoExists) {
        return res.status(400).json({ message: "Roll number already exists" });
      }
    }

    // Update fields
    if (name) student.name = name;
    if (email) student.email = email;
    if (rollNo) student.rollNo = rollNo;
    if (department) student.department = department;
    if (year) student.year = year;
    if (fatherName) student.fatherName = fatherName;
    if (motherName) student.motherName = motherName;
    if (contactNumber) student.contactNumber = contactNumber;
    if (address !== undefined) student.address = address;
    if (dateOfBirth !== undefined) {
      student.dateOfBirth = dateOfBirth && dateOfBirth.trim() !== "" ? dateOfBirth : undefined;
    }
    if (gender !== undefined) student.gender = gender;
    if (photo !== undefined) student.photo = photo;
    if (collegeIdCard !== undefined) student.collegeIdCard = collegeIdCard;
    if (status) student.status = status;

    await student.save();

    res.json({
      message: "Student updated successfully",
      student,
    });
  } catch (error) {
    console.error("Error updating student:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

// Delete student (admin only)
router.delete("/:id", verifyToken, isAdmin, async (req, res) => {
  try {
    const student = await Student.findById(req.params.id);
    if (!student) {
      return res.status(404).json({ message: "Student not found" });
    }

    await Student.findByIdAndDelete(req.params.id);

    res.json({ message: "Student deleted successfully" });
  } catch (error) {
    console.error("Error deleting student:", error);
    res.status(500).json({ message: "Server error", error: error.message });
  }
});

export default router;