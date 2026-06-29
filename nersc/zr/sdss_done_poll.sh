set +e
J=55062964
host=perlmutter
until [ -z "$(ssh $host "squeue -j $J -h -o %i" 2>/dev/null | grep $J)" ]; do sleep 180; done
echo "job $J left queue at $(date)"
mkdir -p /tmp/zr_plots/sdss
scp -q $host:'/pscratch/sd/j/joe2k/sdss_ft/out/*.png' /tmp/zr_plots/sdss/ 2>/dev/null
scp -q $host:'/pscratch/sd/j/joe2k/sdss_ft/out/metrics_*.json' /tmp/zr_plots/sdss/ 2>/dev/null
echo "=== run.log tail ==="
ssh $host 'tail -40 /pscratch/sd/j/joe2k/sdss_ft/run.log'
echo "=== local pngs ==="
ls -la /tmp/zr_plots/sdss/
echo POLL_DONE
