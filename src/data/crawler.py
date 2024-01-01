import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import csv
import os.path


def crawl_data(movieid, imdbid, tmdbid):
    # 发送 GET 请求获取网页内容
    try:
        url = f'https://www.imdb.com/title/tt{imdbid}/'
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/119.0.0.0"
                          "Safari/537.36"
        }
        # 重复5次至成功
        response = None
        for i in range(5):
            response = requests.get(url=url, headers=headers)
            if response.status_code == 200:
                break

        soup = BeautifulSoup(response.content, 'html.parser')

        # 评论数
        view_tab = soup.find(attrs={"data-testid": 'reviewContent-all-reviews'})

        critic_num = 0
        critic_tab = None
        view_num = 0
        reviews_num = 0
        reviews_tab = None
        metascore_tab = None
        metascore = 0
        if view_tab:
            all_reviews_tab = view_tab.find('span', class_='three-Elements')

            if all_reviews_tab:
                reviews_tab = all_reviews_tab.find(class_='score')
                critic_tab = all_reviews_tab.find_next('span', class_='three-Elements').find(class_='score')

                metascore_tab = critic_tab.find_next('span', class_='three-Elements').find(class_='score')
            else:
                all_reviews_tab = view_tab.find('span', class_='less-than-three-Elements')
                if all_reviews_tab:
                    reviews_tab = all_reviews_tab.find(class_='score')
                else:
                    print(movieid + 'reviews-error')
                critic = all_reviews_tab.find_next('span', class_='less-than-three-Elements')
                if critic:
                    critic_tab = critic.find(class_='score')
                else:
                    print(movieid + 'critic_reviews_error')

            view_num = reviews_tab.get_text()
            # 处理特殊符号
            if 'K' in view_num:
                reviews_num = int(float(view_num.replace('K', '')) * 1000)
            else:
                reviews_num = int(view_num)

            if critic_tab:
                critic_num = critic_tab.get_text()
            if metascore_tab:
                metascore = metascore_tab.get_text()
            else:
                print(movieid + 'metascore_error')
        else:
            print(movieid + 'view-error')

        # 票房
        BoxOffice = soup.find(attrs={'data-testid': 'BoxOffice'})
        budget = 0
        domestic_gross = 0
        opening_gross = 0
        global_gross = 0
        if BoxOffice:
            # 预计
            budget_tab = BoxOffice.find(attrs={'data-testid': 'title-boxoffice-budget'})
            if budget_tab:
                budget_num = budget_tab.find('span', class_='ipc-metadata-list-item__list-content-item').get_text()
                budget_numbers = re.findall(r'\d+', budget_num)
                budget = int(''.join(budget_numbers))  # 将列表中的数字部分连接成字符串，并转换为整数
            else:
                print(movieid + 'budget-error')

            # US&Canada
            domestic_gross_tab = BoxOffice.find(attrs={'data-testid': 'title-boxoffice-grossdomestic'})

            if domestic_gross_tab:
                domestic_gross_num = domestic_gross_tab.find('span',
                                                             class_='ipc-metadata-list-item__list-content-item').get_text()
                domestic_numbers = re.findall(r'\d+', domestic_gross_num)
                domestic_gross = int(''.join(domestic_numbers))  # 将列表中的数字部分连接成字符串，并转换为整数

            else:
                print(movieid + "domestic-error")
            # opening
            opening_gross_tab = BoxOffice.find(attrs={'data-testid': 'title-boxoffice-openingweekenddomestic'})
            if opening_gross_tab:
                opening_gross_num = opening_gross_tab.find('span',
                                                           class_='ipc-metadata-list-item__list-content-item').get_text()
                opening_numbers = re.findall(r'\d+', opening_gross_num)
                opening_gross = int(''.join(opening_numbers))  # 将列表中的数字部分连接成字符串，并转换为整数

            else:
                print(movieid + 'opening-error')
            # global
            global_gross_tab = BoxOffice.find(attrs={'data-testid': 'title-boxoffice-cumulativeworldwidegross'})
            if global_gross_tab:
                global_num = global_gross_tab.find('span',
                                                   class_='ipc-metadata-list-item__list-content-item').get_text()
                global_numbers = re.findall(r'\d+', global_num)
                global_gross = int(''.join(global_numbers))  # 将列表中的数字部分连接成字符串，并转换为整数
            else:
                print(movieid + 'global_gross-error')
        else:
            budget = 0
            domestic_gross = 0
            opening_gross = 0
            global_gross = 0
            print(movieid + 'gross-error')

        # runtime
        total_minutes = 0
        time_tab = soup.find(attrs={'data-testid': 'title-techspec_runtime'})
        if time_tab:
            time_str = time_tab.find('div')
            time_s = time_str.get_text()
            pattern = r'(\d+)\s+(hour|hours|minute|minutes)'
            matches = re.findall(pattern, time_s)
            total_minutes = 0
            for match in matches:
                value, unit = match
                value = int(value)

                # 根据单位进行转换
                if unit.startswith('hour'):
                    total_minutes += value * 60
                elif unit.startswith('minute'):
                    total_minutes += value
        else:
            print(movieid + 'runtime-error')
        # rating
        rating = 0
        rating_tab = soup.find(attrs={'data-testid': 'hero-rating-bar__aggregate-rating__score'})
        if rating_tab:
            rating = rating_tab.find('span').get_text()
        else:
            print(movieid + 'rating-error')

        movie_data = {'movieId': movieid,
                      'reviews': reviews_num, 'critic_reviews': critic_num, 'metascore': metascore, 'Budget': budget,
                      'US_and_Canada_Gross': domestic_gross,
                      'Opening_Weekend': opening_gross, 'Global_Gross': global_gross, 'Runtime': total_minutes,
                      'imdb_rating': rating}
        return movie_data
    except Exception as e:
        print('movieId ' + movieid + ' error: ' + str(e))
        return {}


def crawl_links(file_name, output_file):
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            imdb_id = row['imdbId']
            movie_id = row["movieId"]
            tmdb_id = row['tmdbId']
            movie_info = crawl_data(movie_id, imdb_id, tmdb_id)
            df = pd.DataFrame([movie_info])
            file_exists = os.path.isfile(output_file)
            if file_exists:
                mode = 'a'  # 如果文件已经存在，使用追加模式
            else:
                mode = 'w'  # 如果文件不存在，使用写模式

            # 写入到 info.csv 文件，使用追加模式 ('a' 模式)
            df.to_csv(output_file, mode=mode, header=not file_exists, index=False)
