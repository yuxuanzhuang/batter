#!/bin/bash

existing=($(ls -d */FINISHED 2>/dev/null | sed -E 's@z([0-9]+)/FINISHED@\1@' | sort -n))

# Determine range
min=${existing[0]}
max=${existing[-1]}

# Loop and check for missing
for i in $(seq -w $min $max); do
    if [[ ! " ${existing[@]} " =~ " $i " ]]; then
        echo "Missing: z$i/FINISHED"
    fi
done