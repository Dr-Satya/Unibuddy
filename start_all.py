#!/usr/bin/env python3
"""
UniBuddy - Start All Services
This script starts all required services for the UniBuddy application
"""

import os
import sys
import time
import subprocess
import platform

def print_colored(text, color='white'):
    """Print colored text"""
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, colors['white'])}{text}{colors['reset']}")

def check_directory():
    """Check if running in correct directory"""
    if not os.path.exists('package.json'):
        print_colored("Error: Please run this script from the UniBuddy root directory", 'red')
        sys.exit(1)

def start_service(name, command, cwd=None):
    """Start a service in a new terminal window"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows
        if cwd:
            full_command = f'cd {cwd} && {command}'
        else:
            full_command = command
        subprocess.Popen(['start', 'cmd', '/k', full_command], shell=True)
    elif system == 'Darwin':
        # macOS
        if cwd:
            full_command = f'cd {cwd} && {command}'
        else:
            full_command = command
        subprocess.Popen(['osascript', '-e', f'tell app "Terminal" to do script "{full_command}"'])
    else:
        # Linux
        if cwd:
            full_command = f'cd {cwd} && {command}'
        else:
            full_command = command
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{full_command}; exec bash'])

def main():
    """Main function"""
    required = ["FRONTEND_URL", "AUTH_URL", "BACKEND_URL"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    frontend_url = os.environ["FRONTEND_URL"].strip()
    auth_url = os.environ["AUTH_URL"].strip()
    chatbot_url = os.environ["BACKEND_URL"].strip()

    print_colored("Starting UniBuddy Services...", 'cyan')
    print()
    
    # Check directory
    check_directory()
    
    print_colored("Starting services...", 'yellow')
    print()
    
    # Start Auth Backend
    print_colored("Starting Authentication Backend on Port 5000...", 'cyan')
    start_service(
        "Auth Backend",
        "npm start",
        "backend/Unibuddy-Auth/Unibuddy-MERN-Authentication/backend"
    )
    time.sleep(3)
    
    # Start Chatbot Backend
    print_colored("Starting Chatbot Backend on Port 9000...", 'cyan')
    start_service(
        "Chatbot Backend",
        "python run.py",
        "backend"
    )
    time.sleep(3)
    
    # Start Frontend
    print_colored("Starting Frontend on Port 5173...", 'cyan')
    start_service(
        "Frontend",
        "npm run dev",
        None
    )
    
    print()
    print_colored("All services started!", 'green')
    print()
    print_colored("Access Points:", 'yellow')
    print(f"   Frontend:        {frontend_url}")
    print(f"   Auth Backend:    {auth_url}")
    print(f"   Chatbot Backend: {chatbot_url}")
    print()
    print_colored("Tip: Close all terminal windows to stop all services", 'cyan')
    print()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("Startup cancelled", 'yellow')
        sys.exit(0)
    except Exception as e:
        print_colored(f"Error: {e}", 'red')
        sys.exit(1)
