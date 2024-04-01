#!/bin/bash

# This script was written by Gözde Cilingir; goezde.cilingir@uzh.ch  
# If you use or modify this script, please provide appropriate credit.
# Original script available at [URL or Path to the Script Repository].
# [Additional notes or acknowledgments if applicable]


# Define input and output file names, and minimum distance
input_file="SO_2x_all.sites"
output_file="SO_2x_thinned_1K.sites"
min_distance=1000

# Use AWK to process the data
awk -v min_distance="$min_distance" '
    # For each line in the input file after the first line (header)
    NR > 1 {
        # If the first column (snp ID) is not already in snp_data array
        if (!($1 in snp_data)) {
            # Store the snp ID and its position in snp_data array
            snp_data[$1] = $2
            # Move to the next line
            next
        }
        
        # If the distance between the current position and the last stored position
        # for the same snp ID is greater than or equal to the minimum distance
        if ($2 - snp_data[$1] >= min_distance) {
            # Print the snp ID and its position
            print $1 "\t" $2
            # Update the last stored position for this snp ID
            snp_data[$1] = $2
        }
    }
' "$input_file" > "$output_file"
