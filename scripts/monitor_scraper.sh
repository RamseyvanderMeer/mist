#!/bin/bash
# Monitor BMWFault scraper and restart if needed
# Run this with cron every hour: 0 * * * * /mnt/external_ssd/mist/scripts/monitor_scraper.sh

LOG_FILE="/mnt/external_ssd/mist/scraper_monitor.log"
PID_FILE="/tmp/bmwfault_scraper.pid"

echo "[$(date)] Checking scraper status..." >> $LOG_FILE

# Check if scraper is running
if pgrep -f "fetch_bmwfault_mappings.py" > /dev/null; then
    echo "[$(date)] Scraper is running" >> $LOG_FILE
    
    # Check progress
    cd /mnt/external_ssd/mist
    python3 -c "
import sqlite3
conn = sqlite3.connect('data/databases/mist_data.db')
cursor = conn.execute('SELECT last_pcode FROM bmwfault_fetch_checkpoint WHERE id = 1')
checkpoint = cursor.fetchone()[0]
cursor = conn.execute('SELECT COUNT(DISTINCT pcode) FROM bmwfault_pcodes')
unique = cursor.fetchone()[0]
conn.close()
print(f'Checkpoint: {checkpoint}, Fetched: {unique}')
" >> $LOG_FILE 2>&1
else
    echo "[$(date)] Scraper NOT running - restarting..." >> $LOG_FILE
    
    # Restart scraper
    cd /mnt/external_ssd/mist
    export DATAIMPULSE_PROXY="http://53fea9b7d785faf4c5fb:255c9453c29585cc@gw.dataimpulse.com:823"
    export CAPSOLVER_API_KEY="CAP-D5D9683A36418C5C20815B38A7F4CA4161792A1A187038F922D7B5EABEB2A8E9"
    
    nohup python3 -u scripts/fetch_bmwfault_mappings.py --no-diagview 2>&1 >> bmwfault_auto_$(date +%Y%m%d_%H%M%S).log &
    NEW_PID=$!
    echo $NEW_PID > $PID_FILE
    echo "[$(date)] Restarted scraper with PID: $NEW_PID" >> $LOG_FILE
fi
