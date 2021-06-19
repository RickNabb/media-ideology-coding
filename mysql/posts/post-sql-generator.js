/**
 * Post SQL Generator
 * This file should be run with NodeJS to read a directory of JSON
 * files -- FB posts -- and generate mySQL that will seed a mySQL table
 * with the appropriate data.
 */

const fs = require('fs');
const path = require('path');

const POST_TABLE_NAME = 'posts';
const POST_ACCOUNTS_TABLE_NAME = 'post_accounts';
const post_accounts_ids = {};
const POST_TYPES = {
  'facebook': 1,
  'twitter': 2
};

const main = (args) => {
  const dir = args[2];
  fs.readdir(dir, (err, files) => {
    if (err) throw err;

    const dataFiles = files.filter(file => file.indexOf('.json') > -1);

    // Create a SQL file to seed the post accounts table with appropriate
    // account entries
    const accountSeedSql = dataFilesToAccountSQL(dataFiles);
    fs.writeFile(path.join(dir, 'post_accounts_seed.sql'), accountSeedSql, err => {
      if (err) throw err;
      else console.log(`Wrote post account seed file successfully`);
    });

    // Loop through existing .json data files in the given directory
    dataFiles.map(file => {
      fs.readFile(path.join(dir, file), (err, data) => {
        if (err) throw err;

        const account = file.replace('.json','');

        // Convert each to its equivalent SQL generator script
        const sql = postsJsonToSQL(account, data);

        // Write the new SQL file
        const outFile = `${file.replace('.json','')}.sql`;
        fs.writeFile(path.join(dir, outFile), sql, (err) => {
          if (err) throw err;
          else {
            console.log(`Wrote ${outFile} successfully`);
          }
        });
      });
    });
  });
}

/**
 * Create a mySQL file to seed the post_accounts table with the accounts we're
 * reading data from in our directory. Also update the global account_id map
 * so when writing other files, we can use the right ids.
 * @param {Array} files An array of filenames from a directory to read data from.
 * These will be used to seed account entries in the post_accounts table.
 */
const dataFilesToAccountSQL = files => {
  let sql = '';
  files.map((file, i) => {
    const account = file.replace('.json','');
    const line = `INSERT INTO \`${POST_ACCOUNTS_TABLE_NAME}\` (\`account_name\`) VALUES ("${account}");`;
    sql += `${line}\n`;
    post_accounts_ids[account] = i+1;
  });
  return sql;
}

/**
 * Convert social media post JSON into a mySQL script that will generate
 * the same data into a table.
 * @param {Number} account The account that posted these posts.
 * @param {Object} data JSON data representing a series of social media posts
 * to be converted into mySQL.
 */
const postsJsonToSQL = (account, data) => {
  const json = JSON.parse(data);
  let sql = '';
  json.filter(post => /COVID|coronavirus|virus|covid-19/i.test(`${post.text} ${post.shared_text}`))
    .map(post => {
      let line = 'INSERT INTO `' + POST_TABLE_NAME + '` (`post`, `native_id`, `post_account_id`, `image_url`, `video_url`, `likes`, `num_comments`, `shares`, `time_posted`, `link`, `post_type`, `link_text`) VALUES ('
      line += `"${post['post_text'].replace(/\"/g,'\'')}","${post['post_id']}",${post_accounts_ids[account]},"${post['image']}","${post['video']}",${post['likes']},${post['comments']},${post['shares']},"${post['time']}","${post['link']}",${POST_TYPES['facebook']},"${post['shared_text'].replace(/\"/g,'\'')}");`
      sql += `${line}\n`;
    });
  return sql;
}

main(process.argv)
