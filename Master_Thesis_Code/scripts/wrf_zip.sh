#!/bin/bash

get_namelist_variable () {
    # requires that the working directory is the 
    # wrf output directory
    # $1 : name of variable to be matched
    local variable=$(
        grep --max-count=1 "$1" namelist.input | 
        sed -r "s/$1 *= *(.*,)+/\1/" | # Find the value after the equals sign
        # remove quotes and commas, the g at the end means global replacement
        sed -e "s/'//g" -e 's/"//g' -re "s/, +/ /g" -e "s/,/ /g" |
        sed -r "s/^ //" # Remove leading space if any
    )

    echo "$variable"
}

get_nth_value() {
    # utility function to treat strings with spaces as lists
    # it is able to take the string from std in or command line
    # $1 : string to parse
    # $2 : index to be retrieved, starts at 1

    # Check for too many inputs
    if (( ${#} >2 )) ;  then
        echo "Too many inputs have been given"
        exit 1
    fi

    # assign string_to_search and index
    if (( ${#} < 2 )) ; then    # if there are less than two outputs get string from std in

        if [ -p /dev/stdin ]; then # if data was piped
            read in
            local string_to_search="$in"
            local index=$1
        else
            echo "Not enough inputs given"
            exit 1
        fi

    else                        # Else the string has been provided
        local string_to_search=$1
        local index=$2
    fi

    # obtain value
    echo $( echo "$string_to_search" | cut -d " " -f $index )
}

# all of the results need to be paths to wrf directories
echo "Begin compression of wrf results..."

launch_dir=$( pwd )
declare -a domains=("01" "02" "03")
for path in "$@"
do
    cd $launch_dir
    echo "Zipping $path..."
    if [ ! -d $path ] ; then
        echo "$path is not a directory"
        exit 1
    fi
    cd $path
    wrfout_naming_pattern=$( get_namelist_variable "history_outname" )
    start_year=$( get_namelist_variable "start_year" | get_nth_value 3 )
    start_month=$( get_namelist_variable "start_month" | get_nth_value 3)
    start_day=$( get_namelist_variable "start_day" | get_nth_value 3)

    for domain in "${domains[@]}" 
    do  
        echo "Zipping domain $domain..."
        mkdir -p "download_data/d${domain}_zipped"
        wrfout_full_domain_pattern=$( 
            echo $wrfout_naming_pattern | 
            sed "s/<domain>/${domain}/" | # remove domain keyword
            sed "s/<date>/\*:00/"    | # replace date keyword with 20* so we can glob them
            sed -e "s~/~~" -e "s~^.~~" 
        )

        wrfout_full_domain=$( ls . | grep -Eo $wrfout_full_domain_pattern )

        # split files
        echo "Splitting full domain file..."
        split_name=$( echo $wrfout_full_domain | sed -re "s~_[0-9]{2}:[0-9]{2}:[0-9]{2}~_full_domain~" -e "s~.nc~-split.nc.~" )
        split -b 2000000k "./$wrfout_full_domain" "$split_name"
        
        # gzip split files
        echo "Zipping split files..."
        gzip -f "${split_name}"*

        # move files to correct location
        echo "Moving split .gz files..."
        mv ${split_name}*.gz "download_data/d${domain}_zipped/"

        # zip any extracted files
        wrfout_extracted_pattern=$( 
            echo $wrfout_naming_pattern | 
            sed "s/<domain>/${domain}/" | # remove domain keyword
            sed "s/<date>/${start_year}-${start_month}-${start_day}_[a-zA-Z]\*/"    # replace date keyword with 20* so we can glob them
        )
        
        echo "Zipping any extracted files..."
        gzip -k $wrfout_extracted_pattern
        mv $wrfout_extracted_pattern.gz  "download_data/d${domain}_zipped/"

    done


    
    echo "Begin compression of radar results..."
    tar -zcf download_data/crsim_outputs.tar.gz crsim*
    cp "namelist.input" "download_data/"
    
done
