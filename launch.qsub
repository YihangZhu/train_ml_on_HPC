#!/bin/bash

gpus=1
nodes=1
world_size=1
for i in {0..0}; do
  echo "#!/bin/bash" > qsub.submission
  echo "#PBS -q gpu"
  echo "#PBS -l walltime=48:00:00" >> qsub.submission  ## maximum runtime of the job is 72 hours
  echo "#PBS -l pvmem=32gb" >> qsub.submission         ## memory per process, should be at least 32g when using GPUs
  echo "#PBS -l nodes=1:ppn=28:gpus=${gpus}" >> qsub.submission ## maximum 2 GPU per node, each GPU is controlled by one process, ppn is the processors (cores) per node
  echo "#PBS -N qsub_job_${i}" >> qsub.submission              ## name of the job, can be set to whatever you like
  echo "#PBS -m bea" >> qsub.submission # begin end abort send email notification
  echo "#PBS -M <yz681@leicester.ac.uk>" >> qsub.submission
  echo "#PBS -o ic_${i}.log" >> qsub.submission   ## output file for normal log
  echo "#PBS -e ic_${i}.log" >> qsub.submission  ## output file for errors

  #module load cuda90/cudnn/7.0
  #module load cuda90/toolkit/9.0.176
  #module load python/gcc/3.9.10
  #python --version
  echo "source /lustre/ahome3/y/yz681/py36/bin/activate" >> qsub.submission
  echo "python --version"
  echo "cd /lustre/ahome3/y/yz681/ic/">> qsub.submission
  # https://github.com/NVlabs/stylegan2-ada-pytorch/issues/190
  #pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  #pip install -r requirements.txt
  echo "ip1=\`hostname -I | awk '{print \$1}'\`">> qsub.submission

  echo "echo \$ip1">> qsub.submission
  echo "python main.py --cfg config/cifar/Res32_cifar10_1.yaml --cfg_specify config/specify/cifar_imbalance_exp_0-01.yaml --nodes ${nodes} --gpus ${gpus} --world_size ${world_size} --ip \$ip1 --port 40706 --wks 12 --mode ${mode}" >> qsub.submission
  cat qsub.submission
  qsub qsub.submission
  rm qsub.submission
done