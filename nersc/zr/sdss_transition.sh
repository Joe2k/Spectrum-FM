sleep 300
ssh perlmutter 'echo "=== pretok done? ==="; grep -E "merged|=== \[3" /pscratch/sd/j/joe2k/sdss_ft/run.log 2>/dev/null | tail -5
echo "=== first finetune run (expect world=4) ==="; grep -E "\[ft\]|shots=|world=|--- v3|Error|Traceback|OutOfMemory|RuntimeError" /pscratch/sd/j/joe2k/sdss_ft/run.log 2>/dev/null | tail -15' 2>&1 | grep -vE "NOTICE|Laboratory|property|expectation|intercepted|authorized|recording|Unauthorized|disciplinary|stated|consent|^\*|Login|^$|monitored|enforcement|By using|Any or all|conditions"
