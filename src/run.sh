#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=qwen2.5_1.5B_breast_cancer_abs_100_ep10   #Set the job name to "JobExample4"
#SBATCH --time=2:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=48G                     #Request 16GB per node 
#SBATCH --output=qwen2.5_1.5B_breast_cancer_abs_100_ep10_out.%j   #Set the output file name to "JobExample4.out"
#SBATCH --gres=gpu:a100:1          #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132705883597          #Set billing account to 123456
#SBATCH --mail-type=ALL              #Send email on all job events
#SBATCH --mail-user=hasnat.md.abdullah@tamu.edu    #Send all emails to email_address 


cd /scratch/group/instalearn/InstaLearn/src

#finetune dataset parent dir
# finetune_datasets=/scratch/group/instalearn/InstaLearn/src/finetuning_datasets
# #if the dir not available then create it, else do nothing
# if [ ! -d "$finetune_datasets" ]; then
#     mkdir -p $finetune_datasets
# else
#     echo "Directory $finetune_datasets already exists."
# fi
 
# fine_dataset_path=${finetune_datasets}/fine_dataset_100k_abstracts
# previous run settings
# python finetune_pipeline.py --abstract_count 100000 \
# --evaluate_student \
# --finetune_dataset_path  $fine_dataset_path \
# --train_student \
# --epochs 5

# python finetune_pipeline.py --start_idx 0 --end_idx 10000 \
# --evaluate_student

#done
# python finetune_pipeline.py --start_idx 10000 --end_idx 20000 --evaluate_student

#done - pending
# python finetune_pipeline.py --start_idx 20000 --end_idx 30000 --evaluate_student

#done
# python finetune_pipeline.py --start_idx 30000 --end_idx 40000 --evaluate_student

#done
# python finetune_pipeline.py --start_idx 40000 --end_idx 50000 --evaluate_student

#done
# python finetune_pipeline.py --start_idx 50000 --end_idx 60000 --evaluate_student

#done - pending
# python finetune_pipeline.py --start_idx 60000 --end_idx 70000 --evaluate_student

#done - pending
# python finetune_pipeline.py --start_idx 70000 --end_idx 80000 --evaluate_student

#done - pending
# python finetune_pipeline.py --start_idx 80000 --end_idx 90000 --evaluate_student

#done - pending
# python finetune_pipeline.py --start_idx 90000 --end_idx 100000 --evaluate_student


# run 10k abstract with 3B student model + 10 epoch
# fine_dataset_path=/scratch/group/instalearn/InstaLearn/src/finetuning_datasets/fine_dataset_0_100k_abstract
# python finetune_pipeline.py --finetune_dataset_path $fine_dataset_path --train_student --epochs 2


# fine_dataset_path=/scratch/group/instalearn/InstaLearn/src/finetuning_datasets/fine_dataset_10k_abstracts
# python finetune_pipeline.py --finetune_dataset_path  $fine_dataset_path --train_student --epochs 10

# python _finetune_pipeline.py /scratch/group/instalearn/InstaLearn/configs/qwen25_1.5B_exp.yaml

# python pubmed_breast_cancer_finetune_pipeline.py /scratch/group/instalearn/InstaLearn/configs/llama3.2_1B_breast_cancer.yaml
python pubmed_breast_cancer_finetune_pipeline.py /scratch/group/instalearn/InstaLearn/configs/qwen25_1.5B_breast_cancer.yaml