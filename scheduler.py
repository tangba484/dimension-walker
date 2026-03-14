from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from zoneinfo import ZoneInfo

from services.stock_service import refresh_all_saved_tickers

HOUR = 7
MINUTE = 0

def scheduled_job():
    print("===== 매일 자동 갱신 작업 시작 =====")
    refresh_all_saved_tickers()
    print("===== 매일 자동 갱신 작업 종료 =====")


if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone=ZoneInfo("Asia/Seoul"))

    scheduler.add_job(
        scheduled_job,
        trigger=CronTrigger(hour=HOUR, minute=MINUTE),
        id="daily_stock_refresh",
        replace_existing=True,
        max_instances=1
    )

    print(f"스케줄러 실행 중... (매일 한국시간 {HOUR:02d}:{MINUTE:02d})")
    scheduler.start()