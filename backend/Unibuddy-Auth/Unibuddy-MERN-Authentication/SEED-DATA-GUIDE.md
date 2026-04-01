# 🌱 Seed Data Guide

## Overview
This guide explains how to populate your database with sample student data for testing the Admin Panel.

---

## ✅ Successfully Added Students

The following 4 students have been added to your database:

### 1. Rahul Sharma
- **Roll No**: 2024CS001
- **Email**: rahul.sharma@gdgu.org
- **Department**: Computer Science
- **Year**: 3rd Year
- **Father**: Vijay Sharma
- **Mother**: Priya Sharma
- **Contact**: 9876543210
- **DOB**: May 15, 2003
- **Gender**: Male
- **Status**: ACTIVE

### 2. Priya Patel
- **Roll No**: 2024CS002
- **Email**: priya.patel@gdgu.org
- **Department**: Computer Science
- **Year**: 2nd Year
- **Father**: Ramesh Patel
- **Mother**: Sunita Patel
- **Contact**: 9876543211
- **DOB**: August 22, 2004
- **Gender**: Female
- **Status**: ACTIVE

### 3. Amit Kumar
- **Roll No**: 2024ME001
- **Email**: amit.kumar@gdgu.org
- **Department**: Mechanical Engineering
- **Year**: 4th Year
- **Father**: Suresh Kumar
- **Mother**: Anita Kumar
- **Contact**: 9876543212
- **DOB**: December 10, 2002
- **Gender**: Male
- **Status**: ACTIVE

### 4. Sneha Reddy
- **Roll No**: 2024EC001
- **Email**: sneha.reddy@gdgu.org
- **Department**: Electronics & Communication
- **Year**: 1st Year
- **Father**: Venkat Reddy
- **Mother**: Lakshmi Reddy
- **Contact**: 9876543213
- **DOB**: March 18, 2005
- **Gender**: Female
- **Status**: ACTIVE

---

## 🎯 View Students in Admin Panel

1. **Login as Admin**
   - Go to http://localhost:5173/login
   - Login with your admin account

2. **Navigate to Admin Panel**
   - Go to http://localhost:5173/admin
   - Or click "Admin Panel" in navigation

3. **View Students**
   - You should see all 4 students in the table
   - Statistics should show: Total: 4, Active: 4

---

## 🔄 Re-running the Seed Script

If you want to add more students or re-run the script:

```bash
npm run seed
```

**Note**: The script will skip students that already exist (based on email or roll number).

---

## 🧪 Testing Admin Panel Features

Now that you have sample data, test these features:

### 1. Search Functionality
- Search for "Rahul" - should find Rahul Sharma
- Search for "Computer Science" - should find 2 students
- Search for "2024CS001" - should find Rahul Sharma

### 2. Filter by Status
- Filter: All - shows 4 students
- Filter: Active - shows 4 students
- Filter: Inactive - shows 0 students
- Filter: Suspended - shows 0 students

### 3. Pagination
- Currently showing all 4 on page 1
- Add more students to test pagination (10 per page)

### 4. Edit Student
- Click edit icon on any student
- Change status to "INACTIVE"
- Save and verify statistics update

### 5. Delete Student
- Click delete icon on any student
- Confirm deletion
- Verify student is removed and statistics update

### 6. Export to CSV
- Click "Export" button
- Check downloads folder for CSV file
- Open file to verify data

### 7. Add New Student
- Click "Add Student"
- Fill form with new data
- Submit and verify student appears in table

---

## 📊 Sample Test Scenarios

### Scenario 1: Department-wise Search
1. Search "Computer Science" → 2 results
2. Search "Mechanical" → 1 result
3. Search "Electronics" → 1 result

### Scenario 2: Year-wise Filter
1. Add filter by year (if implemented)
2. Year 1 → 1 student (Sneha)
3. Year 2 → 1 student (Priya)
4. Year 3 → 1 student (Rahul)
5. Year 4 → 1 student (Amit)

### Scenario 3: Status Management
1. Edit Rahul → Change to INACTIVE
2. Statistics: Active: 3, Inactive: 1
3. Filter by Inactive → Shows Rahul
4. Edit back to ACTIVE

### Scenario 4: Bulk Operations
1. Export all students → CSV with 4 records
2. Delete 1 student → 3 remaining
3. Add 2 new students → 5 total
4. Export again → CSV with 5 records

---

## 🗑️ Clear All Sample Data

If you want to remove all sample students:

### Method 1: Using MongoDB Compass
1. Open MongoDB Compass
2. Connect to your database
3. Navigate to `students` collection
4. Delete documents with rollNo starting with "2024"

### Method 2: Using MongoDB Shell
```javascript
// Connect to your database
use UnibuddyAuthDB

// Delete sample students
db.students.deleteMany({
  rollNo: { $in: ["2024CS001", "2024CS002", "2024ME001", "2024EC001"] }
})
```

### Method 3: Using Admin Panel
1. Go to Admin Panel
2. Click delete icon on each student
3. Confirm deletion

---

## 📝 Adding More Sample Data

To add more students, edit `backend/fixIndexAndSeed.js`:

```javascript
const sampleStudents = [
  // ... existing students ...
  {
    name: "New Student Name",
    email: "newstudent@gdgu.org",
    rollNo: "2024XX999",
    department: "Department Name",
    year: "1",
    fatherName: "Father Name",
    motherName: "Mother Name",
    contactNumber: "9876543214",
    address: "Address",
    dateOfBirth: new Date("2005-01-01"),
    gender: "Male",
    status: "ACTIVE"
  }
];
```

Then run:
```bash
npm run seed
```

---

## 🔧 Troubleshooting

### Issue: Duplicate Key Error
**Error**: `E11000 duplicate key error`

**Solution**: Student with same email or roll number already exists. Either:
- Use different email/roll number
- Delete existing student first
- Script will automatically skip duplicates

### Issue: Validation Error
**Error**: `Email must be a valid GDGU email`

**Solution**: Ensure all emails end with `@gdgu.org`

### Issue: Connection Error
**Error**: `MongoServerError: connect ECONNREFUSED`

**Solution**: 
- Ensure MongoDB is running
- Check MONGO_URI in .env file
- Verify network connectivity

---

## 🎓 Sample Data Statistics

After seeding:
- **Total Students**: 4
- **Active**: 4
- **Inactive**: 0
- **Suspended**: 0
- **Departments**: 3 (CS, ME, EC)
- **Years**: All 4 years represented
- **Gender**: 2 Male, 2 Female

---

## 🚀 Next Steps

1. ✅ Students added successfully
2. ✅ View in Admin Panel
3. ✅ Test all CRUD operations
4. ✅ Test search and filter
5. ✅ Test export functionality
6. ✅ Add more students as needed

---

**Your Admin Panel is now populated with sample data and ready for testing!** 🎉
