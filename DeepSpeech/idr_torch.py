#!/usr/bin/env python
# coding: utf-8
 
import os
import hostlist
 
# get node list from slurm
hostnames = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])
MASTER_ADDR = hostnames[0]
 
# get SLURM variables
rank = int(os.environ['SLURM_PROCID'])
local_rank = int(os.environ['SLURM_LOCALID'])
size = int(os.environ['SLURM_NTASKS'])
cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
 
# define MASTER_ADD & MASTER_PORT
os.environ['MASTER_ADDR'] = MASTER_ADDR
os.environ['MASTER_PORT'] = '12345'
