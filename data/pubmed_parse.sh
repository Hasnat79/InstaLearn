xml_dir=/scratch/group/instalearn/InstaLearn/data/raw_xml_files/pubmed_data
output_dir=/scratch/group/instalearn/InstaLearn/data/csv_files
for xml_file in "$xml_dir"/*.xml; do
  output_path="$output_dir/$(basename "$xml_file").csv"
  echo "Processing $xml_file"
  python pubmed_csv_parser.py $xml_file --all --output "$output_path"
done

# split_path=/scratch/group/instalearn/InstaLearn/data/pubmed25n0001.xml
# output_path=
# python pubmed_csv_parser.py --input $split_path --output $split_path.csv