#!/bin/bash

# Start the first script in the background and measure its time
(time python experiments/measurement_notes/traditional_logistic_regression.py > output1.txt 2>&1) & pid1=$!

# Start the second script in the background and measure its time
(time python experiments/measurement_notes/traditional_random_forest.py > output2.txt 2>&1) & pid2=$!

# Start the third script in the background and measure its time
(time python experiments/measurement_notes/traditional_xgboost.py > output3.txt 2>&1) & pid3=$!

# Wait for all background processes to complete
wait $pid1 $pid2 $pid3
