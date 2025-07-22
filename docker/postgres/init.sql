-- PostgreSQL initialization script for Predictive Maintenance System
-- This script will be executed when the PostgreSQL container starts for the first time

-- Create database if it doesn't exist (this is handled by POSTGRES_DB environment variable)
-- But we can add any additional setup here

-- Set timezone
SET timezone = 'UTC';

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- You can add any additional database initialization here
-- For example, creating additional users, setting permissions, etc.

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Predictive Maintenance Database initialized successfully';
END $$;
