import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import cookieParser from "cookie-parser";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../.env') });

import { connectDB } from "./db/connectDB.js";

import authRoutes from "./routes/auth.route.js";
import studentRoutes from "./routes/student.route.js";
import userRoutes from './routes/user.routes.js';

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: process.env.CLIENT_URL, credentials: true }));

app.use(express.json({ limit: '50mb' })); // Increased limit for base64 images
app.use(express.urlencoded({ limit: '50mb', extended: true }));
app.use(cookieParser());

app.use("/api/auth", authRoutes);
app.use("/api/students", studentRoutes);
// Add this with your other routes
app.use('/api/users', userRoutes);

app.get("/", (req, res) => {
  res.send("UniBuddy Backend is Live 🚀");
});


app.listen(PORT, () => {
  connectDB();
  console.log("Server is running on port no: ", PORT);
  console.log("MongoDB URI:", process.env.MONGO_URI?.substring(0, 50) + "...");
});