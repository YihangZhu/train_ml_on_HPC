#!/bin/bash
nodes=1
gpus=1
world_size=1
mode='train'
setup=('')
for i in {0..10}; do
  echo "#!/bin/bash" > script.slurm
  echo "#SBATCH --job-name=slurm_${i}" >> script.slurm
  echo "#SBATCH --mail-user=yz681@leicester.ac.uk" >> script.slurm
  echo "#SBATCH --mail-type=ALL" >> script.slurm
  echo "#SBATCH --output=ic_${i}.err" >> script.slurm
  echo "#SBATCH --nodes=${nodes} ## request n nodes" >> script.slurm
  echo "#SBATCH --ntasks-per-node=1 # total number of times running train.py" >> script.slurm
  echo "#SBATCH --cpus-per-task=42 ## each GPU contain 128 cpus (workers), each GPU is a task." >> script.slurm
  echo "#SBATCH --mem-per-cpu=3850" >> script.slurm
  echo "#SBATCH --gres=gpu:ampere_a100:${gpus} # maximum 3 GPUs per node." >> script.slurm
  echo "#SBATCH --partition=gpu" >> script.slurm
  echo "#SBATCH --time=48:00:00  # maximum 48 hours" >> script.slurm
  echo "#SBATCH --account=xxxx  #budget account of leicester university" >> script.slurm


  echo "module load GCCcore/11.3.0" >> script.slurm
  echo "module load Python/3.10.4" >> script.slurm
  echo "source /gpfs/home/y/yz681/code/py3_10_4/bin/activate" >> script.slurm
  echo "python --version" >> script.slurm

  echo "cd /gpfs/home/y/yz681/code/ic/" >> script.slurm

  echo "port=\$(expr 10000 + \$(echo -n \$SLURM_JOBID | tail -c 4))" >> script.slurm

  #echo $MASTER_PORT
  #### get the first node name as master address - customized for vgg slurm
  #### e.g. master(gnodee[2-5],gnoded1) == gnodee2
  #echo "NODELIST="${SLURM_NODELIST}
  #master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  #export MASTER_ADDR=$master_addr
  #echo "MASTER_ADDR="$MASTER_ADDR

  # https://github.com/NVlabs/stylegan2-ada-pytorch/issues/190
  #pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
  #pip install -r requirements.txt
  echo "ip1=\`hostname -I | awk '{print \$1}'\`" >> script.slurm

  echo "echo \$ip1" >> script.slurm

  # srun will run main.py in each node independently, each is associated with a unique number os.environ['SLURM_PROCID'],
  # which can be read in python, used as node rank.
  echo "srun python main.py --cfg config/cifar/Res32_cifar10_1.yaml --nodes ${nodes} --gpus ${gpus} --world_size ${world_size} --ip \$ip1 --port \$port --wks 12 --mode ${mode}" >> script.slurm

  cat script.slurm
  sbatch script.slurm
  rm script.slurm
done
