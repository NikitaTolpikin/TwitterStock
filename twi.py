from birdy.twitter import UserClient
from datetime import datetime, timedelta
import pytz

CONSUMER_KEY = 'LidfFLGcW1NLM2IojLHUDSuw2'
CONSUMER_SECRET = 'kr4utbqPLFI8AhOvA6NtGHHD4Si3pAmY5FC9WcyE7PpSzq0Rzx'
ACCESS_TOKEN = '2260381756-XyYtQFslBXzmlJh6SqMd4VIZ6twKI5SRDM2CUBF'
ACCESS_TOKEN_SECRET ='Sg0vaaNo7aDgXMzyrfaSJtFdv6hGlThHw3uM6jjSHQp1L'

client = UserClient(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_TOKEN_SECRET)

response = client.api.users.show.get(screen_name='Gazprom')
tw_count = response.data['statuses_count']
tweets = []
k=0
while tw_count > 0:
    if k == 0:
        response = client.api.statuses.user_timeline.get(screen_name='Gazprom', count=tw_count, tweet_mode='extended')
        k=1
    else:
        response = client.api.statuses.user_timeline.get(screen_name='Gazprom', count=tw_count, tweet_mode='extended', max_id=maxid)
    tw_count -= 200
    single_tweets = response.data
    print(single_tweets[0])
    for item in single_tweets:
        tweets.append(item)
    maxid = single_tweets[-1].id

i=0
for tweet in tweets:
    print(i)
    i+=1
    print(tweet.full_text)
    d = datetime.strptime(tweet.created_at,'%a %b %d %H:%M:%S %z %Y').replace(tzinfo=pytz.UTC) + timedelta(hours=3)
    print(d.strftime('%Y-%m-%d %H:%M:%S'))
    
