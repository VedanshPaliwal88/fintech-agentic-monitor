import schedule
import time
import logging
from datetime import datetime
from orchestrator import AgentOrchestrator, SETTINGS

# --- Logger Setup ---
# Configure to log to a file for background execution
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_monitoring_job():
    """
    The job function to be scheduled.
    """
    logger.info("=" * 50)
    logger.info("Scheduler starting the daily monitoring job.")
    try:
        monitor = AgentOrchestrator()
        result = monitor.run_monitoring_cycle()
        logger.info(f"Monitoring job finished. Result: {result}")
    except Exception as e:
        logger.error("An unhandled exception occurred during the scheduled run:", exc_info=True)
    logger.info("=" * 50)


if __name__ == "__main__":
    # --- Schedule the job ---
    # You can change the schedule here, e.g., schedule.every(4).hours, etc.
    schedule_time = "09:00"
    schedule.every().day.at(schedule_time).do(run_monitoring_job)

    logger.info(f"Scheduler started. Monitoring job is scheduled daily at {schedule_time}.")
    logger.info("Running the job once now to ensure everything is working...")
    run_monitoring_job()

    while True:
        schedule.run_pending()
        time.sleep(60) # Check every minute if it's time to run