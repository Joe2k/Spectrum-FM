#!/bin/bash
SSH="ssh -o IdentitiesOnly=yes -o LogLevel=ERROR -o ConnectTimeout=25 -i $HOME/.ssh/nersc joe2k@perlmutter"
CK=/pscratch/sd/j/joe2k/deepsrch/checkpoints/abmask_dec_zv2
LOG=/tmp/zr_plots/dec_monitor.log; mkdir -p /tmp/zr_plots
echo "=== dec monitor start $(date) ===" >> $LOG
while true; do
  OUT=$($SSH "J=\$(squeue -u joe2k -h -o '%j' | grep -cx abdec); S=\$(grep -oE '\"step\": [0-9]+' $CK/metrics.jsonl 2>/dev/null | grep -oE '[0-9]+' | sort -n | tail -1); echo \"job=\$J step=\${S:-0}\"" 2>/dev/null)
  J=$(echo "$OUT" | grep -oE 'job=[0-9]+' | cut -d= -f2); S=$(echo "$OUT" | grep -oE 'step=[0-9]+' | cut -d= -f2)
  J=${J:-0}; S=${S:-0}
  echo "[$(date +%H:%M:%S)] job=$J step=$S" >> $LOG
  if [ "$S" -ge 50000 ] 2>/dev/null; then echo "DEC_DONE step=$S"; exit 0; fi
  if [ "$J" = "0" ] 2>/dev/null; then echo "DEC_STALLED step=$S (abdec gone, <50k) — needs resume"; exit 2; fi
  sleep 300
done
