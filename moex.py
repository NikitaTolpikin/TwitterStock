import asyncio
from datetime import datetime, timedelta
import aiomoex
import pandas as pd


async def main():
    async with aiomoex.ISSClientSession():
        data = await aiomoex.get_market_candles(share, interval=1, start=start_time, end=end_time)
        df = pd.DataFrame(data)
        return df


def get_candles(share_name, start_date, end_date):
    global share
    global start_time
    global end_time
    start_time = start_date.strftime('%Y-%m-%d %H:%M')
    end_time = end_date.strftime('%Y-%m-%d %H:%M')
    share = share_name
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(main())
