// data API, receives the JSON data from Nicla Sense ME and writes it 
// to the CSV file, which is created firing the samplefile API.

var express = require('express');
var router = express.Router();
const fs = require('fs');
var global_vars = require('../routes/globals'); // global vars module
var path = require('path');
// Json to CSV Converter options
const options = {
  delimiter: {
    field: ';',
    eol: '\n',
  },
  trimHeaderFields: true,
    prependHeader: false
}
// Create the Json to CSV converter
const converter = require('json-2-csv');

/* GET data page. */
router.get('/', function (req, res) {
  res.send("This API is intended for POST method only");
});

/* POST data page. */
router.post('/', function (req, res, next) {

  // convert the JSON array in CSV
  converter.json2csvAsync(req.body, options).then(csv => {

    // append CSV to the file, whose name comes from globals.js
    if (global_vars.fName != "") {
      csvRecord = csv + ";" + global_vars.tagName + "\n";
      fs.appendFileSync(path.resolve(process.cwd() + '/data/' + global_vars.fName), csvRecord);
      res.send("Data written to CSV file ");
    }
    else {
      console.log("Sample file undefined");
      res.send("Sample file undefined");
    }
  }).catch(err => console.log(err));
});

module.exports = router;
