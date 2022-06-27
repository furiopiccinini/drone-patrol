/**
  samplefile.js module receives the sample file name and create it adding the full path and
  the header 
  @module samplefile.js
*/

/** Require declarations */
var express = require('express');
var router = express.Router();
var dateTime = require('../dateTime.js'); /** dateTime module, to create timestamp @module dateTime*/
var globalVars = require('./globals'); /** global vars module */
const fs = require('fs');
var path = require('path');

/** GET control for unsupported calls */
router.get('/', function (req, res) {
  res.send('GET method not accepted');
});

/** POST create file name with full path */
router.post('/', function (req, res) {

  try {
    /** Set the TAG name added to any sampled record */
    globalVars.tagName = req.body.filename;
    /** Create the file appending date, time and file extension */
    globalVars.fName = globalVars.tagName + '-' + dateTime.dateTime() + ".csv";
    console.log("File name: " + globalVars.fName);
    res.send("File name created");
  }
  catch (error) {
    console.log(error);
  }
});

module.exports = router;
