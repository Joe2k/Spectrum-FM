#!/bin/bash
# Local (mac-side) monitor for the V4 run. Polls NERSC; relaunches the next 4h
# chunk (v4_once.sh, resume-aware + singleton-guarded) whenever no v4zv2 job is
# queued and step<200000. Exits when step>=200000 (then the agent runs the eval).
SSH="ssh -o IdentitiesOnly=yes -o LogLevel=ERROR -o ConnectTimeout=25 -i $HOME/.ssh/nersc joe2k@perlmutter"
CK=/pscratch/sd/j/joe2k/deepsrch/checkpoints/approach_a_v2cache_x2x3_zv2_ddp4
LOG=/tmp/zr_plots/v4_monitor.log
mkdir -p /tmp/zr_plots
TARGET=200000
echo "=== v4 monitor start $(date) ===" >> $LOG
while true; do
  OUT=$($SSH "J=\$(squeue -u joe2k -h -o '%j' | grep -cx v4zv2); S=\$(grep -oE '\"step\": [0-9]+' $CK/metrics.jsonl 2>/dev/null | tail -1 | grep -oE '[0-9]+'); echo \"job=\$J step=\${S:-0}\"" 2>/dev/null)
  J=$(echo "$OUT" | grep -oE 'job=[0-9]+' | cut -d= -f2)
  S=$(echo "$OUT" | grep -oE 'step=[0-9]+' | cut -d= -f2)
  J=${J:-0}; S=${S:-0}
  echo "[$(date +%H:%M:%S)] job=$J step=$S" >> $LOG
  if [ "$S" -ge "$TARGET" ] 2>/dev/null; then
    echo "[$(date)] REACHED $TARGET — monitor exiting for eval" >> $LOG
    echo "V4_DONE step=$S"
    break
  fi
  if [ "$J" = "0" ]; then
    echo "[$(date)] no v4zv2 job — relaunching chunk" >> $LOG
    $SSH "cd /pscratch/sd/j/joe2k/v4; nohup bash v4_once.sh > once_\$(date +%s).log 2>&1 & disown" 2>/dev/null
    sleep 60
  fi
  sleep 600
done
echo "=== v4 monitor end $(date) ===" >> $LOG
