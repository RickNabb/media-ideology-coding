/*
Set up the mySQL tables for the covid coding app.
*/

CREATE TABLE IF NOT EXISTS article_accounts (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  account_name VARCHAR(256) NOT NULL
);

/* HOWARD MASK-WEARING CODES */
CREATE TABLE IF NOT EXISTS articles_mask (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post VARCHAR(10240),
  native_id TEXT,
  article_account_id BIGINT
);

CREATE TABLE IF NOT EXISTS articles_training (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post VARCHAR(10240),
  native_id TEXT,
  article_account_id BIGINT,
  training_index INT
);

CREATE TABLE IF NOT EXISTS mask_wearing_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  attribute VARCHAR(10240),
  code INT DEFAULT -1,
  confidence INT,
  session_id BIGINT,
  timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mask_wearing_training_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  attribute VARCHAR(10240),
  code INT DEFAULT -1,
  confidence INT,
  session_id BIGINT,
  timestamp TIMESTAMP
);

CREATE TABLE IF NOT EXISTS coding_settings (
  distribution_method VARCHAR(256),
  outlets_distributed VARCHAR(256)
);