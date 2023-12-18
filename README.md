# create ray cluster
d3x ray create -n sdiff-ft -i dkubex123/sdiff:rayft --cpu 3  --gpu 1 --memory 15  --hcpu 8 --hmemory 16 --type g4dn.xlarge --htype r6i.8xlarge

# Run finetuning script
./finetune.sh

# It generates a lora adapter in the output directory
