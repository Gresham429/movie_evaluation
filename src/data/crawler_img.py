import requests
from bs4 import BeautifulSoup
import csv
import os.path

def scraper_data(movieid, imdbid, tmdbid, external_folder_path):
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
        for _ in range(5):
            response = requests.get(url=url, headers=headers)
            if response.status_code == 200:
                break

        if response.status_code != 200:
            print('movieId' + movieid + ': error')
            return

        soup = BeautifulSoup(response.content, 'html.parser')
        img_tab = soup.find(attrs={'data-testid': 'hero-media__poster'})
        if img_tab:
            img_url = img_tab.find('img').get('src')
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                folder_path = external_folder_path + '\\img'
                file_name = f"{movieid}.png"
                # 保存图片到文件夹中
                with open(os.path.join(folder_path, file_name), 'wb') as f:
                    f.write(img_response.content)
                    f.close()
                    print('movieId: ' + movieid + ' file saved')
            else:
                print('error')
    except Exception as e:
        print('movieId' + movieid + ': error', e)

def crawl_img(links_file, external_folder_path):
    # 读取csv文件
    with open(links_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            imdb_id = row['imdbId']
            id = row["movieId"]
            tmdb_id = row['tmdbId']
            scraper_data(id, imdb_id, tmdb_id, external_folder_path)
