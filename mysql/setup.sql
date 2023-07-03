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
  article_account_id BIGINT,
  post_type BIGINT,
  CONSTRAINT fk_article_mask_account
  FOREIGN KEY (article_account_id)
    REFERENCES article_accounts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
);

CREATE TABLE IF NOT EXISTS articles_training (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post VARCHAR(10240),
  native_id TEXT,
  post_account_id BIGINT,
  post_type BIGINT,
  training_index INT,
  CONSTRAINT fk_article_mask_account
  FOREIGN KEY (article_account_id)
    REFERENCES article_accounts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
);

CREATE TABLE IF NOT EXISTS mask_wearing_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  code INT DEFAULT -1,
  confidence INT,
  session_id BIGINT,
  CONSTRAINT fk_mask_wearing_post
  FOREIGN KEY (article_id)
    REFERENCES articles_mask(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mask_wearing_training_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  code INT DEFAULT -1,
  confidence INT,
  session_id BIGINT,
  CONSTRAINT fk_mask_wearing_post
  FOREIGN KEY (article_id)
    REFERENCES articles_training(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mask_wearing_disagree_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_mask_wearing_disagree_post
  FOREIGN KEY (article_id)
    REFERENCES articles_mask(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mask_wearing_agree_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_mask_wearing_agree_post
  FOREIGN KEY (article_id)
    REFERENCES articles_mask(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS sentence_highlight_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  article_id BIGINT,
  sentence_id VARCHAR(256),
  code INT DEFAULT -1,
  CONSTRAINT fk_sentence_highlight_post
  FOREIGN KEY (article_id)
    REFERENCES articles_mask(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);