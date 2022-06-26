var dateTime = () => {
    // create a Date object and get all details
    var date = new Date();
    return date.getDate().toString() + "-" + 
                    (date.getMonth()+1).toString() + "-" + 
                    date.getFullYear().toString() + "_" + 
                    (date.getHours()+1).toString() + "-" + 
                    (date.getMinutes()+1).toString() + "-" + 
                    (date.getSeconds() + 1).toString();
}

module.exports = { dateTime };