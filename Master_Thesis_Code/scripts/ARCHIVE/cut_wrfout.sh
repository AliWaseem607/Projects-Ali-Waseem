#!/bin/bash

# Code to repeated slice a file into smaller chunks
# $1 file to split

#######################################################
# Help command
#######################################################
Help () {
    echo "usage: bash cut_wrfout.sh [Options] wrfout"
    echo
    echo "Options"
    echo
}

#######################################################
# Command line Parsing
#######################################################

expected_args=1
set -e

# Check for correct number of arguments
if [ $# -ne $expected_args ]; then
    echo "Invalid number of arguments."
    echo "Run bash cut_wrfout.sh --help for more information"
    exit 1
fi

WRFInput=$1
split_name=$( echo $WRFInput | sed "s~.nc~-split.nc.~" )
split -b 2000000k "$WRFInput" "$split_name"

