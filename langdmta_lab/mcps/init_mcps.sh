mkdir -p logs

cd utility
python server.py > ../logs/utility.log 2>&1 & pid_utility=$!

cd ../design
python server.py > ../logs/design.log 2>&1 & pid_design=$!

cd ../synthesis
python server.py > ../logs/synthesis.log 2>&1 & pid_synthesis=$!

echo "Servers started. PIDs: $pid_utility $pid_design $pid_synthesis"
echo "Press Ctrl+C to stop."

trap 'echo "Stopping..."; kill $pid_utility $pid_design $pid_synthesis 2>/dev/null || true' INT TERM
wait
