#!/bin/bash

# Install Node.js and npm (for Ubuntu/Debian)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install prettier globally
sudo npm install -g prettier@3.4.2

# Create data directory with proper permissions
sudo mkdir -p /data
sudo chmod 777 /data
sudo mkdir -p /data/logs /data/docs
sudo chmod -R 777 /data

# Create a test markdown file
echo "# Test\n\n* Item 1\n*    Item 2" > /data/format.md
chmod 666 /data/format.md 