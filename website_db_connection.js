/**
 * A script to interface between the website and any other program.
 * Author: Nick Rabb (nicholas.rabb@tufts.edu)
 */

const mysql = require('mysql');
const fs = require('fs');
const path = require('path');

const WEBSITE_HOST = '216.172.184.57';
const WEBSITE_USER = 'nickrabb_root';
const WEBSITE_PW = 'wB08&pKQWwXb';
const WEBSITE_DB = 'nickrabb_covid_coding';

const LABEL_DATA_DIR = './labeled-data';

const CODE_TABLES = [
  'ambiguity',
  'complexity',
  'dangerous_virus',
  'disgust',
  'get_virus',
  'group',
  'morality',
  'purity',
  'spread_virus',
  'threat',
];

const websiteConn = () => {
  return mysql.createConnection({
    host: WEBSITE_HOST,
    user: WEBSITE_USER,
    password: WEBSITE_PW,
    database: WEBSITE_DB
  });
}

/**
 * Connect to the website DB and pull down all the labeled data to write to
 * external files. These files can then be used to augment data.
 */
const retrieveLabelData = () => {
  const conn = websiteConn();
  conn.connect(err => {
    if (err) throw err;
    CODE_TABLES.map(tableName => {
      conn.query(`SELECT ${tableName}_codes.code as code, posts.native_id as native_id FROM ${tableName}_codes JOIN posts ON ${tableName}_codes.post_id = posts.id`, (err, res) => {
        if (err) throw err;
        const resJson = res.map(row => ({
          code: row.code,
          native_id: row.native_id
        }));
        fs.writeFile(path.join(LABEL_DATA_DIR, `${tableName}.json`), JSON.stringify(resJson), err => {
          if (err) throw err;
          console.log(`Wrote ${tableName}.json successfully.`);
        });
      });
    });
  });
}

retrieveLabelData();