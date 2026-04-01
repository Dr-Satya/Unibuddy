import express from 'express';
const router = express.Router();
import { 
  getAllUsers, 
  createUser, 
  updateUser, 
  deleteUser 
} from '../controllers/user.controller.js';
import { verifyToken } from '../middleware/verifyToken.js';

// Check if user is admin
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

// All routes require admin authentication
router.get('/', verifyToken, isAdmin, getAllUsers);
router.post('/', verifyToken, isAdmin, createUser);
router.put('/:id', verifyToken, isAdmin, updateUser);
router.delete('/:id', verifyToken, isAdmin, deleteUser);

export default router;