from birdy.twitter import UserClient
from datetime import datetime, timedelta
import pytz
import moex as mx
import csv


CONSUMER_KEY = 'LidfFLGcW1NLM2IojLHUDSuw2'
CONSUMER_SECRET = 'kr4utbqPLFI8AhOvA6NtGHHD4Si3pAmY5FC9WcyE7PpSzq0Rzx'
ACCESS_TOKEN = '2260381756-XyYtQFslBXzmlJh6SqMd4VIZ6twKI5SRDM2CUBF'
ACCESS_TOKEN_SECRET = 'Sg0vaaNo7aDgXMzyrfaSJtFdv6hGlThHw3uM6jjSHQp1L'

client = UserClient(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_TOKEN,ACCESS_TOKEN_SECRET)
field_name_1 = 'id'
field_name_2 = 'text'
field_name_3 = 'mark'

marked = []
all = []

AO = ['GAZP', 'LKOH', 'MGNT', 'ROSN', 'SBER', 'GMKN', 'TATN', 'TRNFP', 'NVTK', 'NLMK', 'HYDR', 'FEES', 'URKA', 'MAGN',
      'IRAO', 'BANE', 'MTSS', 'UELM', 'MFON', 'MVID', 'NKNC', 'CHEP', 'PHOR', 'KMAZ', 'UTAR']
TW = ['Gazprom', 'lukoilrus', 'magnit_info', 'RosneftRu', 'sberbank', 'NornikOfficial', 'tatneftofficial', 'TransneftRu',
      'novatek_', 'nlmk', 'rushydro_', 'FSK_EES', 'uralkali_russia', 'MMK_Steel', 'interrao', 'BashneftMedia', 'ru_mts',
      'copper_elem', 'megafonru', 'mvideo', 'nknhprom', 'ChelPipe', 'phosagro', 'kamaz_official', 'UTairAvia']

k = 0
index = 0
m_index = 0
while k < len(AO):
    while True:
        try:
            response = client.api.users.show.get(screen_name=TW[k])
            break
        except:
            pass
    tw_count = response.data['statuses_count']
    if tw_count > 3200:
        tw_count = 3200
    tweets = []

    first = 0
    while tw_count > 0:
        if first == 0:
            response = client.api.statuses.user_timeline.get(screen_name=TW[k], count=tw_count,
                                                             tweet_mode='extended', exclude_replies=True)
            first = 1
        else:
            while True:
                try:
                    response = client.api.statuses.user_timeline.get(screen_name=TW[k], count=tw_count,
                                                             tweet_mode='extended', max_id=maxid, exclude_replies=True)
                    break
                except:
                    pass
        tw_count -= 200
        single_tweets = response.data
        for item in single_tweets:
            tweets.append(item)
        maxid = single_tweets[-1].id

    print(len(tweets))
    for tweet in tweets:
        tw_text = tweet.full_text
        all.append({field_name_1: index, field_name_2: tw_text})
        index += 1
        d = datetime.strptime(tweet.created_at, '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=pytz.UTC) + timedelta(hours=3)
        print(index)
        print(tw_text)
        if d.hour > 19 or d.hour < 10:
            continue
        while True:
            try:
                df = mx.get_candles(AO[k], d, d + timedelta(minutes=5))
                break
            except:
                pass
        if not df.empty:
            flag = 0
            df_values = df.values
            for item in df_values:
                percent = (item[1]-item[-3])/(item[1])*100
                if abs(percent) > 0.10:
                    if percent > 0:
                        flag = 1
                    elif percent < 0:
                        flag = -1
                if flag != 0:
                    break
            marked.append({field_name_1: m_index, field_name_2 : tw_text, field_name_3: flag})
            m_index += 1
            print("Влияние: ", flag)

    with open('all.csv', 'a', encoding='UTF-8') as f:
        w = csv.DictWriter(f, fieldnames=[field_name_1, field_name_2])
        if k == 0:
            w.writeheader()
        for data in all:
            w.writerow(data)

    with open('marked.csv', 'a', encoding='UTF-8') as f:
        w = csv.DictWriter(f, fieldnames=[field_name_1, field_name_2, field_name_3])
        if k == 0:
            w.writeheader()
        for data in marked:
            w.writerow(data)
    marked = []
    all = []
    k += 1
