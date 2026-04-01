import mongoose from "mongoose";
import dotenv from "dotenv";
import { Student } from "./models/student.model.js";

dotenv.config();

const sampleStudents = [
  {
    name: "Rahul Sharma",
    email: "rahul.sharma@gdgu.org",
    rollNo: "2024CS001",
    department: "Computer Science",
    year: "3",
    fatherName: "Vijay Sharma",
    motherName: "Priya Sharma",
    contactNumber: "9876543210",
    address: "123 MG Road, Delhi",
    dateOfBirth: new Date("2003-05-15"),
    gender: "Male",
    status: "ACTIVE"
  },
  {
    name: "Priya Patel",
    email: "priya.patel@gdgu.org",
    rollNo: "2024CS002",
    department: "Computer Science",
    year: "2",
    fatherName: "Ramesh Patel",
    motherName: "Sunita Patel",
    contactNumber: "9876543211",
    address: "456 Park Street, Mumbai",
    dateOfBirth: new Date("2004-08-22"),
    gender: "Female",
    status: "ACTIVE"
  },
  {
    name: "Amit Kumar",
    email: "amit.kumar@gdgu.org",
    rollNo: "2024ME001",
    department: "Mechanical Engineering",
    year: "4",
    fatherName: "Suresh Kumar",
    motherName: "Anita Kumar",
    contactNumber: "9876543212",
    address: "789 Lake View, Bangalore",
    dateOfBirth: new Date("2002-12-10"),
    gender: "Male",
    status: "ACTIVE"
  },
  {
    name: "Sneha Reddy",
    email: "sneha.reddy@gdgu.org",
    rollNo: "2024EC001",
    department: "Electronics & Communication",
    year: "1",
    fatherName: "Venkat Reddy",
    motherName: "Lakshmi Reddy",
    contactNumber: "9876543213",
    address: "321 Hill Road, Hyderabad",
    dateOfBirth: new Date("2005-03-18"),
    gender: "Female",
    status: "ACTIVE"
  }
];

const fixIndexAndSeed = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(process.env.MONGO_URI);
    console.log("✅ Connected to MongoDB");

    // Get all indexes
    const indexes = await Student.collection.getIndexes();
    console.log("\n📋 Current indexes:", Object.keys(indexes));

    // Drop old rollNumber index if it exists
    try {
      if (indexes.rollNumber_1) {
        await Student.collection.dropIndex("rollNumber_1");
        console.log("✅ Dropped old 'rollNumber_1' index");
      }
    } catch (err) {
      console.log("ℹ️  No old rollNumber index to drop");
    }

    // Ensure correct indexes exist
    await Student.collection.createIndex({ email: 1 }, { unique: true });
    await Student.collection.createIndex({ rollNo: 1 }, { unique: true });
    console.log("✅ Ensured correct indexes are in place");

    console.log("\n🌱 Starting to seed students...\n");

    // Check if students already exist and add them
    let addedCount = 0;
    let skippedCount = 0;

    for (const studentData of sampleStudents) {
      const existingStudent = await Student.findOne({
        $or: [
          { email: studentData.email },
          { rollNo: studentData.rollNo }
        ]
      });

      if (existingStudent) {
        console.log(`⚠️  Student already exists: ${studentData.name} (${studentData.rollNo})`);
        skippedCount++;
        continue;
      }

      // Create new student
      const student = new Student(studentData);
      await student.save();
      console.log(`✅ Added student: ${studentData.name} (${studentData.rollNo})`);
      addedCount++;
    }

    console.log("\n" + "=".repeat(60));
    console.log("🎉 Seeding completed successfully!");
    console.log(`📊 Summary: ${addedCount} added, ${skippedCount} skipped`);
    console.log("📊 You can now view these students in the Admin Panel");
    console.log("🌐 Admin Panel: http://localhost:5173/admin");
    console.log("=".repeat(60));
    
  } catch (error) {
    console.error("\n❌ Error:", error.message);
    if (error.errors) {
      Object.keys(error.errors).forEach(key => {
        console.error(`   - ${key}: ${error.errors[key].message}`);
      });
    }
  } finally {
    // Close connection
    await mongoose.connection.close();
    console.log("\n🔌 Database connection closed");
    process.exit(0);
  }
};

// Run the function
fixIndexAndSeed();
