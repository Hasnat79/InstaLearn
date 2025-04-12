#!/bin/bash

#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=download_pubmed_25      #Set the job name to "JobExample4"
#SBATCH --time=05:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=1        #Request 8 tasks/cores per node
#SBATCH --mem=32G                     #Request 16GB per node
#SBATCH --output=download_pubmed_25.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:rtx:1          #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue




# Script to download PubMed XML files from NCBI FTP server
# Downloads files from pubmed25n0001.xml.gz to pubmed25n1274.xml.gz

# Base URL for the PubMed baseline data
BASE_URL="https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"

# Create a directory to store the downloaded files
mkdir -p pubmed_data

# Log file to track progress
LOG_FILE="download_log.txt"
echo "Download started at $(date)" > $LOG_FILE

# Function to download a file with retry mechanism
download_with_retry() {
    local file_num=$1
    local formatted_num=$(printf "%04d" $file_num)
    local file_name="pubmed25n${formatted_num}.xml.gz"
    local url="${BASE_URL}/${file_name}"
    local max_retries=3
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        echo "Downloading $file_name (Attempt $((retry_count+1)))"
        if wget -q -O "pubmed_data/${file_name}" "$url"; then
            echo "Successfully downloaded $file_name" | tee -a $LOG_FILE
            success=true
        else
            retry_count=$((retry_count+1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Failed to download $file_name. Retrying in 5 seconds..." | tee -a $LOG_FILE
                sleep 5
            else
                echo "Failed to download $file_name after $max_retries attempts." | tee -a $LOG_FILE
            fi
        fi
    done
    
    return $([ "$success" = true ] && echo 0 || echo 1)
}

# Main download loop
#1274
for i in $(seq 1 1274); do
    download_with_retry $i
    
    # Add a small delay between downloads to be nice to the server
    sleep 1
done

# Summary
echo "Download process completed at $(date)" | tee -a $LOG_FILE
echo "Files should be stored in the pubmed_data directory" | tee -a $LOG_FILE

# Count successful downloads
successful_downloads=$(find pubmed_data -name "pubmed25n*.xml.gz" | wc -l)
echo "Successfully downloaded $successful_downloads out of 1274 files" | tee -a $LOG_FILE