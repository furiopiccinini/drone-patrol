# Create the dataset header file from template
cat dataset_template > Neuton_$2.csv
# Append data samples
cat $1*.csv >> Neuton_$2.csv
echo "Dataset created"
