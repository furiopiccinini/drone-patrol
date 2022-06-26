/**
This backend is part of a project related to the Nicla Sense ME board. 
The project is made of a python-programmed drone, which has the Nicla Sense ME board onboard.
When the drone moves as decided by the programmer, the web interface connects with the board 
via BLE Bluetooth, recording data from the onboard sensors. The data is then sent to this server,
with two APIs:

- /samplefile, generates a file name with the timestamp appended
- /data, receives the data sent by the Nicla Sense ME and store it as a CSV file

This project is part of a series that will be published on a book from APress, written by
Enrico Miglino.

Author: Furio Piccinini
Date: May 2022
License: Apache
*/

var createError = require('http-errors');
var express = require('express');
var path = require('path');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var logger = require('morgan');
var cors = require('cors');
var indexRouter = require('./routes/index');
var usersRouter = require('./routes/users');
var dataRouter = require('./routes/data'); // data route
var sampleFileRouter = require('./routes/samplefile'); // samplefile route

var app = express();

// view engine setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');

app.use(cors());
app.use(logger('dev'));
app.use(express.json());
app.use(bodyParser.urlencoded({
  extended: true
}));
app.use(cookieParser());
app.use(express.static(path.join(__dirname, 'public')));

app.use('/', indexRouter);
app.use('/users', usersRouter);
app.use('/data', dataRouter); // data router
app.use('/samplefile', sampleFileRouter); //samplefile router

// catch 404 and forward to error handler
app.use(function (req, res, next) {
  next(createError(404));
});

// error handler
app.use(function (err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};

  // render the error page
  res.status(err.status || 500);
  res.render('error');
});

module.exports = app;
