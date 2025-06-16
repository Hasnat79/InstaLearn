


input_dir=/scratch/group/instalearn/InstaLearn/data/csv_files/
output_dir=/scratch/group/instalearn/InstaLearn/data/breast_cancer_pubmed/
total_files=$(ls -1q $input_dir*.csv | wc -l)
current_file=0

for file in $input_dir*.csv; do
    current_file=$((current_file + 1))
    echo "Processing file: $file ($current_file/$total_files)"
    filename=$(basename "$file")
    output_file="$output_dir/filtered_$filename"
    python ../filter_abstracts_based_on_breast_cancer_mesh.py --input_file "$file" --output_file "$output_file"
done
