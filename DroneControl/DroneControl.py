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
is_debug = False                # Set to true to enable the debug logs
json_file = './dronecontrol.json' # Drone control file
manual_control = False           # bypass the Json file processing and accept terminal commands
# Current command in execution from the "fly" list
cmdIndex = 0
# Flag set to true when the next command should be battery check
isBatteryCheck = False
# IF this flag is set, the fly should be halted immediately forcing landing.
haltFly = False
# Response returned by the drone after a command has been sent
dataDroneResponse = ''
# Flag set when the last command response from drone has been received.
nextCommandReady = True

# Define the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Lists with the drone path commands
# Commands to manage the fly along the desired path.
droneFly = ['']
# Number of times the fly sequence should be repeated
flyLoops = 0
# The battery level is periodically checked to monitor the remaining drone fly time.
batteryLevel = 0
# The minimum battery level before forcing a laning
batteryAlert = 0
# It is used to check the battery level. It is the interval to control the battery.
flyTimeToCheck = 0.0

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
    with open(json_file) as file:
        dictionary = json.load(file)
    
    # Load the list of commands
    droneFly = dictionary['fly']

    # Load limits and configuration parameters
    flyLoops = int(dictionary['loops'])
    flyTimeToCheck = float(dictionary['timeToCheck'])

    logging.info('Loaded dronecontrol Json file completed.')
    logging.info('-- Parameters --\nFly loops: %d\nBattery limit: %d\nTime to check battery: %f', 
                flyLoops, batteryAlert, flyTimeToCheck)
    logging.info('--- Commands to execute %d', len(droneFly))

    
def recvDroneResponse():
    '''
    Wait for a response from the drone after a command is sent.
    The method enable a timeout thread to avoid waiting undefinitely if
    some error occurs (manual stop, crash, signal lost).
    '''    
    global haltFly
    global dataDroneResponse
    global nextCommandReady

    time.sleep(5.0)
    nextCommandReady = True
    
    # while(nextCommandReady == False):
    #     try:
    #         data, server = sock.recvfrom(8890)
    #         dataDroneResponse = data.decode(encoding="utf-8")
    #         print("Response: " + dataDroneResponse)
    #         logging.info('Response - %s', dataDroneResponse)
    #         nextCommandReady = True
    #     except Exception:
    #         logging.error('No or wrong data received from the drone. Fly should stop.')
    #         haltFly = True
    #         nextCommandReady = True
    #         break

def sendDroneCommand(cmd):
    '''
    Send a command to the drone wia UDP and wait for the response
    '''
    time.sleep(1)
    msg = cmd.encode(encoding="utf-8") 
    sent = sock.sendto(msg, tello_address)
    logging.info('API: %s', cmd)
    # Wait to receive a drone response from the paralle thread
    # while(nextCommandReady is False):
    #     time.sleep(0.1)
    #     logging.info("nextCommandRead still false")


def processDroneCommand(cmdIndex, thread):
    '''
    Send a command to the drone wia UDP from the commands processing
    list.
    '''
    sendDroneCommand(droneFly[cmdIndex])


def haltDroneFly():
    '''
    Force drone landing. Program stops.
    '''
    sendDroneCommand("land")
    sock.close()
    logging.info('!!! Land forced on error or low battery')
    exit()


# -----------------------------------------------------------------------------------------------
# Thread functions
# -----------------------------------------------------------------------------------------------

def timerBattery():
    '''
    The drone battery level should be checked periodically. When the timer occurs,
    the battery check flag is set so it is the next command sent to the drone.
    '''
    logging.info('Timeout for battery check')
    global isBatteryCheck
    isBatteryCheck = True


# -----------------------------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------------------------

def main():
    '''
    Main application.
    Enable the connection and if there are no errors, process the Json file and exectues
    path with the drone.
    '''
    global batteryLevel
    global manual_control
    global nextCommandReady

    # Initialize logging
    logging.basicConfig(filename=log_file, level=log_level,
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')

    logging.info('*** DroneControl new session started ***')

    # receiveResponseThread = threading.Thread(target=recvDroneResponse)
    # receiveResponseThread.start()

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
    # Automated fly based on the droncontrol Json file.
    # the process checks for the battery every minute to avoid crashes
    # and set a timeout on the drone response if something wrong occurs
    else:
        logging.info('Json dronecontrol.json file loading')
        # Load the json control file
        loadDroneControl()

        # Start the timer to set the battery status flag every minute
        # timerBatteryThread = threading.Timer(flyTimeToCheck, timerBattery)

        # Activate the SDK APIs on the Drone
        nextCommandReady = False
        sendDroneCommand("command")
        recvDroneResponse()
        # if(haltFly):
        #     haltDroneFly()

        # Check if the initial battery level is too low to start a fly
        # sendDroneCommand("battery?")
        # batteryLevel = 100 # int(dataDroneResponse)        
        # if( haltFly or (batteryLevel <= batteryAlert)):
        #     logging.info('Battery level before takeoff: %d', batteryLevel)
        #     logging.info('Level too low! Fly not started')
        #     haltDroneFly()
        # else:
        #     # Save the current battery level  and restart the battery timer.
        #     logging.info('Battery level before takeoff: %d', batteryLevel)
        #     batteryLevel = 100 # int(dataDroneResponse)        
        #     # timerBatteryThread.start()

        # --------------------------------------------------------------
        # Commands processor
        # --------------------------------------------------------------
        cmdIndex = len(droneFly)
        loopIndex = 0

        # Process the required number of times the whole command set
        while(loopIndex < flyLoops):
            logging.info('--- Executes the command sequence %d of %d',
                            loopIndex + 1, flyLoops)

            cmdPosition = 0             # Initial command in the list.
            # Process all the command in the list
            while(cmdPosition < cmdIndex):
                # ------------------ Process the next command in the queue
                nextCommandReady = False
                processDroneCommand(cmdPosition)
                recvDroneResponse()
                # if(haltFly):
                #     # timerBatteryThread.cancel()
                #     haltDroneFly()

                # ------------------ Is time to check the battery level?
                # if(isBatteryCheck):
                #     sendDroneCommand("battery?")
                #     recvDroneResponse()
                #     batteryLevel = 100 # int(dataDroneResponse)        
                #     if( haltFly or (batteryLevel <= batteryAlert)):
                #         logging.info('Battery level: %d too low to continue flying.', batteryLevel)
                #         # Asks for the time on fly, update the log and stop
                #         sendDroneCommand("time?")
                #         recvDroneResponse()
                #         logging.info('Fly stopped after %d seconds', int(dataDroneResponse))
                #         haltDroneFly()
                #     else:
                #         # Asks for the time on fly, update the log and restart the timer
                #         logging.info('Battery level: %d', batteryLevel)
                #         # Asks for the time on fly, update the log and stop
                #         sendDroneCommand("time?")
                #         recvDroneResponse()
                #         logging.info('Flying time: %d seconds', int(dataDroneResponse))
                #         # Restart the battery timer.
                #         # timerBatteryThread.start()

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
