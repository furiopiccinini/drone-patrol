package main

import (
	"arduino/bhy/webserver"
)

func main() {
	webserverCommand()
}

func webserverCommand() {
	webserver.Execute()
}
