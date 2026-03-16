#!/bin/bash
bb=$1
echo $bb
sed "s/tt/${bb}/g" job-abp-base.gpu > jobcatheter.gpu

sbatch jobcatheter.gpu
