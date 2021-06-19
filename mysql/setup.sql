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

/* HOWARD MASK-WEARING CODES */

-- CREATE TABLE IF NOT EXISTS h_comfort_difficult_breathing_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_comfort_difficult_breathing_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_comfort_too_hot_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_comfort_too_hot_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_efficacy_health_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_efficacy_health_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_efficacy_ineffective_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_efficacy_ineffective_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_access_difficult_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_access_difficult_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_access_expensive_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_access_expensive_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_compensation_stay_away_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_compensation_stay_away_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_inconvenience_remembering_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_inconvenience_remembering_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_inconvenience_hassle_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_inconvenience_hassle_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_appearance_ugly_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_appearance_ugly_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_appearance_weird_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_appearance_weird_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_attention_untrustworthy_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_attention_untrustworthy_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_attention_uncomfortable_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_attention_uncomfortable_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_independence_forced_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_independence_forced_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

-- CREATE TABLE IF NOT EXISTS h_independence_authority_codes (
--   id BIGINT AUTO_INCREMENT PRIMARY KEY,
--   post_id BIGINT,
--   code INT DEFAULT -1,
--   CONSTRAINT fk_h_independence_authority_post
--   FOREIGN KEY (post_id)
--     REFERENCES posts(id)
--     ON DELETE SET NULL
--     ON UPDATE CASCADE
-- );

CREATE TABLE IF NOT EXISTS mask_wearing_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_mask_wearing_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mask_wearing_disagree_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_mask_wearing_disagree_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS mask_wearing_agree_codes (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post_id BIGINT,
  code INT DEFAULT -1,
  CONSTRAINT fk_mask_wearing_agree_post
  FOREIGN KEY (post_id)
    REFERENCES posts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS articles_mask (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  post VARCHAR(10240),
  native_id TEXT,
  post_account_id BIGINT,
  post_type BIGINT,
  CONSTRAINT fk_article_mask_account
  FOREIGN KEY (post_account_id)
    REFERENCES post_accounts(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE,
  CONSTRAINT fk_article_mask_type
  FOREIGN KEY (post_type)
    REFERENCES post_types(id)
    ON DELETE SET NULL
    ON UPDATE CASCADE
);