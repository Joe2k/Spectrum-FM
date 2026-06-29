sleep 240
ssh perlmutter 'echo "=== queue ==="; squeue -u joe2k -o "%i %T %M" 2>/dev/null
echo "=== merge/finetune ==="; grep -E "merged|=== \[3|world=|--- v|step .*sel_nmad|FINAL|Traceback|Error:|OutOfMemory|NCCL|torchrun|RendezvousError" /pscratch/sd/j/joe2k/sdss_ft/run.log 2>/dev/null | tail -20' 2>&1 | grep -vE "NOTICE|Laboratory|property|expectation|intercepted|authorized|recording|Unauthorized|disciplinary|stated|consent|^\*|Login|^$|monitored|enforcement|By using|Any or all|conditions"
