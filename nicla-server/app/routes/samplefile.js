/*
  samplefile.js module receives the sample file name and create it adding the full path and
  the header 
*/

var express = require('express');
var router = express.Router();
var dateTime = require('../dateTime.js'); // dateTime module, to create timestamp
var globalVars = require('./globals'); // global vars module
const fs = require('fs');
var path = require('path');
var global_vars = require('../routes/globals'); // global vars module

// GET control for unsupported calls
router.get('/', function (req, res) {
  res.send('GET method not accepted');
});

// POST create file name with full path */
router.post('/', function (req, res) {

  try {
    // Set the TAG name added to any sampled record
    globalVars.tagName = req.body.filename;
    // Create the file
    globalVars.fName = globalVars.tagName + '-' + dateTime.dateTime() + ".csv";
    console.log("File name: " + globalVars.fName);
    // File header in CSV format
    // File header has been suspended to be able to concatentate multiple sampling
    // into a single data set.
    // headerCSV = "Temperature;Barometer;Humidity;Gas;TAG\n";
    // fs.appendFileSync(path.resolve(process.cwd() + '/data/' + global_vars.fName), headerCSV);
    res.send("File name created");
  }
  catch (error) {
    console.log(error);
  }
});

module.exports = router;
