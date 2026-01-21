#!/bin/bash

if [[ ! -d "KMeans_files" ]]; then
    if [[ ! -e "KMeans_files.zip" ]]; then
        wget --user="ob-cuda" --password="mini-2526Z" "https://pages.mini.pw.edu.pl/~brodkaj/CUDA/KMeans_files.zip"
    fi
    unzip KMeans_files.zip -d KMeans_files
fi

EXEC=../bin/KMeans

DATA_FORMATS=("bin" "txt")
DATA_FORMATS_LONG=("binary" "text")
DATA_FILES=("./KMeans_files/points_5mln_4d_5c.dat" "./KMeans_files/points_5mln_4d_5c.txt")

COMPUTATION_METHODS=("gpu1" "gpu2" "cpu")
OUTPUT_DIRECTORY="test_output_files"
mkdir -p $OUTPUT_DIRECTORY

clean_stdout_file ()
{
    sed -i 's/time: [0-9]\+\.[0-9]\+/time: _:_/g' $1
    sed -i 's/data format:.*$/data format:/' $1
    sed -i 's/Computing on.*$/Computing on/' $1
}

REFERENCE_STDOUT="cleaned_stdout"


cat ./KMeans_files/stdout_5mln_4d_5c.txt | tr -d '\r' | sed 's/execotion/execution/' > $REFERENCE_STDOUT

clean_stdout_file $REFERENCE_STDOUT

TIMES_FILE="times.txt"
rm -f $TIMES_FILE

for METHOD in ${COMPUTATION_METHODS[@]}
do
    for ((i = 0; i < 2; i++)); do

        OUTPUT_FILE="$OUTPUT_DIRECTORY/${METHOD}_${DATA_FORMATS[i]}_output"

        STDOUT_FILE="$OUTPUT_DIRECTORY/${METHOD}_${DATA_FORMATS[i]}_stdout"

        echo "starting $METHOD on ${DATA_FORMATS[i]}"

        $EXEC ${DATA_FORMATS[i]} "$METHOD" ${DATA_FILES[i]} $OUTPUT_FILE > $STDOUT_FILE &&
            echo "done"
        if [[ $? > 0 ]]; then
            echo ""
            echo "failed"
            exit 1
        fi

        cat ./KMeans_files/results_5mln_4d_5c.txt | tr -d '\r' > "cleaned_results"
        diff $OUTPUT_FILE  "cleaned_results"
        rm  "cleaned_results"

        if [[ $? > 0 ]]; then
            echo "Files don't match"
            exit 1
        fi

        OUTPUTTED_DATA_FORMAT=$(grep 'data format' $STDOUT_FILE | grep -Eo '[^ ]+$')

        if [[ ${DATA_FORMATS_LONG[i]} != $OUTPUTTED_DATA_FORMAT ]]; then
            echo "Incorrect data format outputted"
            echo "Expected ${DATA_FORMATS_LONG[i]} got $OUTPUTTED_DATA_FORMAT"
            exit 1
        fi

        echo "$STDOUT_FILE :" >> $TIMES_FILE
        grep 'time: [0-9]\+\.[0-9]\+' $STDOUT_FILE >> $TIMES_FILE
        echo "" >> $TIMES_FILE

        clean_stdout_file $STDOUT_FILE

        diff $STDOUT_FILE $REFERENCE_STDOUT


        if [[ $? > 0 ]]; then
            echo "Stdout doesn't match"
            exit 1
        fi
    done
done

rm $REFERENCE_STDOUT

echo "All tests passed"




