


input_dir=/scratch/group/instalearn/InstaLearn/data/csv_files/
output_dir=/scratch/group/instalearn/InstaLearn/data/cardio_pubmed/
for file in $input_dir*.csv; do
    filename=$(basename "$file")
    output_file="$output_dir/filtered_$filename"
    python ../filter_abstracts_based_on_cardio_mesh.py --input_file "$file" --output_file "$output_file"
done
