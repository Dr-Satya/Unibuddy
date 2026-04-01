import jwt from "jsonwebtoken";
import { User } from "../models/user.model.js";

export const verifyToken = async (req, res, next) => {
	const token = req.cookies.token;

	if (!token) {
		return res.status(401).json({
			success: false,
			message: "Unauthorized - no token provided",
		});
	}

	try {
		const decoded = jwt.verify(token, process.env.JWT_SECRET);

		if (!decoded) {
			return res.status(401).json({
				success: false,
				message: "Unauthorized - invalid token",
			});
		}

		// ✅ KEEP existing feature
		req.userId = decoded.userId;

		// ✅ ADDITION (without breaking anything)
		const user = await User.findById(decoded.userId).select("-password");

		if (!user) {
			return res.status(401).json({
				success: false,
				message: "User not found",
			});
		}

		// ✅ Attach full user object (new feature)
		req.user = user;

		next();
	} catch (error) {
		console.log("Error in verifyToken ", error);
		return res.status(500).json({
			success: false,
			message: "Server error",
		});
	}
};
