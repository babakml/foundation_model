#!/bin/bash

echo "=========================================="
echo "KIT CLUSTER ARCHITECTURE ANALYSIS"
echo "=========================================="

echo ""
echo "1. NODE INFORMATION:"
echo "-------------------"
echo "Hostname: $(hostname)"
echo "Node type: $(hostname | cut -d'n' -f1)"
echo "CPU info:"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|Architecture"
echo ""

echo "2. MEMORY INFORMATION:"
echo "---------------------"
free -h
echo ""

echo "3. AVAILABLE PARTITIONS:"
echo "-----------------------"
sinfo -o "%P %A %D %T %N %G"
echo ""

echo "4. AVAILABLE QOS (Quality of Service):"
echo "-------------------------------------"
sacctmgr show qos format=name,maxwall,maxjobs,maxnodes,priority
echo ""

echo "5. CURRENT JOB LIMITS:"
echo "---------------------"
sacctmgr show user $USER format=user,account,partition,qos,maxjobs,maxwall,maxnodes
echo ""

echo "6. AVAILABLE MODULES:"
echo "--------------------"
module avail 2>&1 | head -20
echo ""

echo "7. GPU INFORMATION (if available):"
echo "---------------------------------"
nvidia-smi 2>/dev/null || echo "No GPUs available on this node"
echo ""

echo "8. STORAGE INFORMATION:"
echo "----------------------"
df -h
echo ""

echo "9. NETWORK INFORMATION:"
echo "----------------------"
echo "Network interfaces:"
ip addr show | grep -E "inet |UP|DOWN" | head -10
echo ""

echo "10. SLURM CONFIGURATION:"
echo "-----------------------"
scontrol show config | grep -E "MaxJobCount|MaxArraySize|MaxNodeCount|DefMemPerCPU|MaxMemPerNode"
echo ""

echo "11. CURRENT QUEUE STATUS:"
echo "------------------------"
squeue -o "%.10i %.9P %.20j %.8u %.8T %.10M %.6D %R" | head -10
echo ""

echo "12. RECENT JOB HISTORY:"
echo "----------------------"
sacct -S $(date -d '1 day ago' +%Y-%m-%d) -u $USER --format=JobID,JobName,Partition,State,ExitCode,Start,End,Elapsed,ReqMem,MaxRSS
echo ""

echo "=========================================="
echo "ANALYSIS COMPLETE"
echo "=========================================="