/**
@summary This backend is part of a project related to the Nicla Sense ME board. 
The project is made of a python-programmed drone, which has the Nicla Sense ME board onboard.
When the drone moves as decided by the programmer, the web interface connects with the board 
via BLE Bluetooth, recording data from the onboard sensors. The data is then sent to this server,
which exposes two APIs:

- /samplefile, generates a file name with the timestamp appended
- /data, receives the data sent by the Nicla Sense ME and store it as a CSV file

In the nicla-server directory you can find the file named "Nicla Server API.postman_collection.json": 
this is the exported collection of all the APIs available in the Node.js server, along with 
an example JSON payload.

This project uses JSDoc as a documentation tool. After installing JSDoc with "npm install -g jsdoc" (global install)
or with "npm install --save-dev jsdoc" (local install as dev dependency), to generate the documentation for the source 
simply cd into the "nicla-server" folder and run "jsdoc app.js". All the documentation will be generated in the "out" folder.
The /data and / samplefile endpoints are located in the "routes" folder, so to generate the docs follow the same procedure in these paths.

This project is part of a series that will be published on a book from APress, written by
Enrico Miglino.

@author: Furio Piccinini
Date: May 2022
@version: 0.8
@license: Apache
*/

/** Require statements */
var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var logger = require('morgan');
/** CORS module to enable Cross Origin Resource Sharing */
var cors = require('cors');
var indexRouter = require('./routes/index');
var dataRouter = require('./routes/data'); // data route
var sampleFileRouter = require('./routes/samplefile'); // samplefile route

var app = express();

/** view engine setup */
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
/** Middleware setup */
app.use(cors());
app.use(logger('dev'));
app.use(express.json());
app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/data', dataRouter); /** data router */
app.use('/samplefile', sampleFileRouter); /** samplefile router */

/** catch 404 and forward to error handler  */
app.use(function (req, res, next) {
  next(createError(404));
});

/** error handler */
app.use(function (err, req, res, next) {
  /** set locals, only providing error in development */
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  /** render the error page */
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
