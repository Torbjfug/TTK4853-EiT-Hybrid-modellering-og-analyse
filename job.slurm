#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --account=ie-itk
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=12000
#SBATCH --job-name="EiT hybrid modelling and analysis"
#SBATCH --output=test-srun.out
#SBATCH --mail-user=gustavoo@stud.ntnu.no
#SBATCH --mail-type=ALL
 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load intel/2018b 
module load Python/3.6.6
source /home/gustavoo/datasci/bin/activate
python /home/gustavoo/geppy/setup.py install
python /home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/part1_equation_discovery/data/generate_data_directions.py
python /home/gustavoo/TTK4853-Eit-Hybrid-modellering-og-analyse/SR/part1_equation_discovery/gep_eq_disc/navier_stokes.py
