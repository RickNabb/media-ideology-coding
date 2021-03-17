/*
Set up the mySQL tables for the covid coding app.
*/

CREATE TABLE IF NOT EXISTS post_accounts (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  account_name VARCHAR(256) NOT NULL
);

CREATE TABLE IF NOT EXISTS post_types (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  `type` VARCHAR(256) NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post VARCHAR(2048),
  image_url VARCHAR(2048),
  video_url VARCHAR(2048),
  post_url VARCHAR(2048),
  likes INT,
  num_comments INT,
  shares INT,
  time_posted DATETIME,
  link VARCHAR(2048),
  link_text VARCHAR(2048),
  native_id TEXT,
  post_account_id BIGINT,
  post_type BIGINT,
  CONSTRAINT fk_post_account
  FOREIGN KEY (post_account_id)
    REFERENCES post_accounts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
  CONSTRAINT fk_post_type
  FOREIGN KEY (post_type)
    REFERENCES post_types(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS complexity_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_complexity_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS ambiguity_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_ambiguity_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS morality_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_morality_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS threat_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_threat_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS disgust_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_disgust_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS purity_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_purity_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS groups (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  group_name VARCHAR(256) NOT NULL
);

CREATE TABLE IF NOT EXISTS group_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  group_id BIGINT,
  CONSTRAINT fk_group_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
  CONSTRAINT fk_group_group
  FOREIGN KEY (group_id)
    REFERENCES groups(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS get_virus_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_get_virus_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS spread_virus_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_spread_virus_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS dangerous_virus_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_dangerous_virus_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);