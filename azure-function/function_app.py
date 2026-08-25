import logging

import azure.functions as func

from buoy_core import update_all

app = func.FunctionApp()


@app.timer_trigger(schedule="0 */30 * * * *", arg_name="mytimer", run_on_startup=False, use_monitor=True)
def buoy_position_timer(mytimer: func.TimerRequest) -> None:
    if mytimer.past_due:
        logging.info("The timer is past due.")
    update_all()
