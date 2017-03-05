#!/bin/bash                                                                                                                                                                            

#SBATCH -n 1                                                                                                                                                                           
#SBATCH -N 1                                                                                                                                                                           
#SBATCH --mem=150000                                                                                                                                                                    


#SBATCH -t 14:00:00 #Indicate duration using HH:MM:SS                                                                                                                                  
#SBATCH -p general #Based on your duration                                                                                                                                             


#SBATCH -o ./training_outputs/output.txt                                                                                                                             
#SBATCH -e ./training_outputs/errs.txt                                                                                                                               
#SBATCH --mail-type=ALL                                                                                                                                                                
#SBATCH --mail-user=malbergo@college.harvard.edu                                                                                                                                       


# --------------                                                                                                                                                                       

source ~/setup1.sh

cd ~/cs181-practicals/p1/
python RF_regression_training.py