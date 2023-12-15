# create ray cluster
d3x ray create -n sdiff-ft -i dkubex123/sdiff:rayft --cpu 12 --gpu 1 --memory 50  --hcpu 2 --hmemory 4 --type g5.4xlarge

# Run finetuning script
./finetune.sh

# It generates a lora adapter in the output directory
