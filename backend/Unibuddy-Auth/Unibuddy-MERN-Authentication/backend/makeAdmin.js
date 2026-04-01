import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import mongoose from "mongoose";
import { User } from "./models/user.model.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../.env') });

const makeAdmin = async (email) => {
  await mongoose.connect(process.env.MONGO_URI);
  
  const user = await User.findOneAndUpdate(
    { email: email.toLowerCase() },
    { role: 'ADMIN' },
    { new: true }
  );

  if (user) {
    console.log(`✅ ${email} is now ADMIN`);
  } else {
    console.log(`❌ User not found: ${email}`);
  }

  await mongoose.disconnect();
  process.exit(0);
};

// Pass email as argument: node makeAdmin.js kiyosha@gdgu.org
const email = process.argv[2];
if (!email) {
  console.log("Usage: node makeAdmin.js <email>");
  process.exit(1);
}

makeAdmin(email);
