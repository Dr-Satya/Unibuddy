#!/usr/bin/env python3
"""
University Assistant AI - Team Management CLI Launcher

This script provides access to the comprehensive team management and 
architectural commands for the University Assistant AI project.
"""

import os
import sys

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from cli.main import main

if __name__ == "__main__":
    print("🚀 University Assistant AI - Team Management CLI")
    print("=" * 50)
    main()
