-- Sample initialization script for NexHacks PostgreSQL database

-- Create a sample table
CREATE TABLE IF NOT EXISTS sample_data (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert some sample data
INSERT INTO sample_data (name, description) VALUES
    ('Sample Item 1', 'This is the first sample item'),
    ('Sample Item 2', 'This is the second sample item'),
    ('Sample Item 3', 'This is the third sample item');

-- Create a users table (without authentication fields)
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample users
INSERT INTO users (username, email) VALUES
    ('demo_user', 'demo@example.com'),
    ('test_user', 'test@example.com');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
