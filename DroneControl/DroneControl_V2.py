#!/usr/bin/python3
#
# Tello Python3 DroneControl application
# For more information check the Tello drone SDK
#
# Important! To make the drone working the computer should be connected
# to the Drone WiFi (drone access point: 192.168.10.1)s
#
# Author: Enrico Miglino <balearicdynamics@gmail.com>
# Version: 0.2
# Date: May 2022

import threading 
import socket
import sys
import time
import logging
import json

# Fixed parameters to connect UPD channel to the drone
host = ''
port = 9000
locaddr = (host,port)
# Socket IP and port are hardware defined by the drone
tello_address = ('192.168.10.1', 8889)

# Other global variables and hardcoded names
log_file = 'dronecontrol.log'   # Log file name
log_level = logging.INFO        # Default log level (maybe INFO, WARNING, ERROR, CRITICAL, DEBUG)
json_file = './dronecontrol.json' # Drone control file
manual_control = False           # bypass the Json file processing and accept terminal commands
# Current command in execution from the "fly" list
cmdIndex = int(0)

# Define the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Lists with the drone path commands
# Commands to manage the fly along the desired path.
droneFly = ['']
# Number of times the fly sequence should be repeated
flyLoops = int(0)

# -----------------------------------------------------------------------------------------------
# Application functions
# -----------------------------------------------------------------------------------------------

def initSocket():
    '''
    Create the UDP socked to send and receive commands to the
    drone via WiFi access
    '''
    # Bind the socket to the host. Local address IP is left empty
    # as the client IP is assigned by the drone via DHCP
    sock.bind(locaddr)
    logging.info('Drone socket bound')


def loadDroneControl():
    '''
    Load the Json drone control file in a dictionary and
    organizes the list
    '''
    global flyLoops
    global droneFly

    with open(json_file) as file:
        dictionary = json.load(file)
    
    # Load the list of commands
    droneFly = dictionary['fly']

    # Load limits and configuration parameters
    flyLoops = int(dictionary['loops'])

    logging.info('Loaded dronecontrol Json file completed.')
    logging.info('--- Commands to execute %d', len(droneFly))

    
def recvDroneResponse():
    '''
    Wait for a response from the drone after a command is sent.
    The method enable a timeout thread to avoid waiting undefinitely if
    some error occurs (manual stop, crash, signal lost).
    '''    
    time.sleep(2.0)

def sendDroneCommand(cmd):
    '''
    Send a command to the drone wia UDP and wait for the response
    '''
    time.sleep(1)
    msg = cmd.encode(encoding="utf-8") 
    sent = sock.sendto(msg, tello_address)
    time.sleep(5.0)

def processDroneCommand(cmdIndex):
    '''
    Send a command to the drone wia UDP from the commands processing
    list.
    '''
    global droneFly
    sendDroneCommand(droneFly[cmdIndex])


# -----------------------------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------------------------

def main():
    '''
    Main application.
    Enable the connection and if there are no errors, process the Json file and exectues
    path with the drone.
    '''
    global manual_control

    # Initialize logging
    logging.basicConfig(filename=log_file, level=log_level,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info('*** DroneControl new session started ***')

    # For testing purposes only, if the manual_control flag is set to true
    # the json control file is ignored and the commands can be sent from the
    # terminal.
    # Remember that the first command to be sent is 'command' to enable the SDK
    # on the drone side.
    if(manual_control):
        logging.info('Manual control is enabled.')
        print ('\r\n\r\n* Manual Drone Control *\r\n')
        print ('Available: command takeoff land flip forward back left right \r\n       up down cw ccw speed speed?\r\n')
        print ('end -- exit the appplication.\r\n')

        receiveResponseThread = threading.Thread(target=recvDroneResponse)
        receiveResponseThread.start()

        while True: 
            try:
                msg = input("");
                
                if not msg:
                    break  
                if 'end' in msg:
                    print ('...')
                    sock.close()  
                    logging.info('End of application. Socket closed')
                    break
                # Send data
                msg = msg.encode(encoding="utf-8") 
                sent = sock.sendto(msg, tello_address)
                logging.info('Manual control. Sent command: %s', msg)
            except KeyboardInterrupt:
                sock.close()  
                logging.info('End on forced keyb interrupt. Socket closed')
                break
    else:
        # Automated fly based on the droncontrol Json file.
        # the process checks for the battery every minute to avoid crashes
        # and set a timeout on the drone response if something wrong occurs
        logging.info('Json dronecontrol.json file loading')
        # Load the json control file
        loadDroneControl()

        # Activate the SDK APIs on the Drone
        sendDroneCommand("command")
        recvDroneResponse()

        # --------------------------------------------------------------
        # Commands processor
        # --------------------------------------------------------------
        cmdIndex = len(droneFly)
        loopIndex = int(0)
        logging.info("Start command processor with %d commands", cmdIndex)
        # Process the required number of times the whole command set
        while(loopIndex < flyLoops):
            logging.info('--- Executes the command sequence %d of %d',
                            loopIndex + 1, flyLoops)

            cmdPosition = int(0)             # Initial command in the list.
            # Process all the command in the list
            while(cmdPosition < cmdIndex):
                # ------------------ Process the next command in the queue
                logging.info("API :" + droneFly[cmdPosition])
                processDroneCommand(cmdPosition)
                recvDroneResponse()
                # Update the command position
                cmdPosition = cmdPosition + 1

            # Update the loop counter
            loopIndex = loopIndex + 1

        logging.info('*** Session closed ***')    

# -----------------------------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    main()
