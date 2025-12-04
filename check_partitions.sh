#!/bin/bash
# Script to check available SLURM partitions on the cluster

echo "Checking available SLURM partitions..."
echo "Run this on the cluster:"
echo ""
echo "# Check available partitions:"
echo "sinfo"
echo ""
echo "# Check partition details:"
echo "sinfo -o \"%P %l %D %T %N\""
echo ""
echo "# Check your account limits:"
echo "sacctmgr show user ul_oqn09"
echo ""
echo "# Check current queue:"
echo "squeue"
