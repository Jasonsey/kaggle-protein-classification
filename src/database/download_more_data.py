import os
import errno
from multiprocessing.pool import Pool as Pool
from tqdm import tqdm
import requests
import pandas as pd
from PIL import Image
from pathlib import Path
from io import BytesIO


def download(pid, image_list, base_url, save_dir, image_size=(512, 512)):
    headers = {
        'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'
    }
    proxies = {
        'http': 'socks5://172.18.39.74:1080',
        'https': 'socks5://172.18.39.74:1080'
    }
    colors = ['red', 'green', 'blue', 'yellow']
    for i in tqdm(image_list, postfix=pid):
        img_id = i.split('_', 1)
        for color in colors:
            img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
            img_name = i + '_' + color + '.png'
            img_url = base_url + img_path
            path = Path(save_dir) / img_name
            if path.exists():
                # print('%s exits' % img_name)
                continue

            # Get the raw response from the url
            try:
                r = requests.get(img_url, allow_redirects=True, stream=True, timeout=50, headers=headers, proxies=proxies)
                # r.raw.decode_content = True
            except:
                print('%s download failed, %s' % (img_url, e))
                continue

            # Use PIL to resize the image and to convert it to L
            # (8-bit pixels, black and white)
            im = Image.open(r.raw)
            im = im.resize(image_size, Image.LANCZOS).convert('L')
            im.save(path, 'PNG')
            # print('New img: %s' % img_name)
            del im
            r.close()

if __name__ == '__main__':
    # Parameters
    process_num = os.cpu_count() * 10
    image_size = (512, 512)
    url = 'http://v18.proteinatlas.org/images/'
    csv_path =  "../../data/input/moredata/HPAv18RBGY_wodpl.csv"
    save_dir = "../../data/input/moredata/train"

    # Create the directory to save the images in case it doesn't exist
    try:
        os.makedirs(save_dir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    print('Parent process %s.' % os.getpid())
    img_list = pd.read_csv(csv_path)['Id']
    list_len = len(img_list)

    while True:
        train_path = Path(save_dir)
        x = 0
        for _ in train_path.glob('*.png'):
            x += 1
        y = 4 * list_len - x
        print('Need more: %s' % y)
        if y <= 0:
            break

        p = Pool(process_num)
        for i in range(process_num):
            start = int(i * list_len / process_num)
            end = int((i + 1) * list_len / process_num)
            process_images = img_list[start:end]
            p.apply_async(
                download, args=(str(i), process_images, url, save_dir, image_size)
            )
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        print('All subprocesses done.')
        


# from gevent import monkey, pool
# monkey.patch_all(thread=False)

# import os

# import requests
# from urllib.request import urlretrieve, urlopen
# import pandas as pd
# import numpy as np
# import gevent
# import os
# from multiprocessing.pool import Pool
# from PIL import Image


# def make_url(i, color):
#     base_url = 'http://v18.proteinatlas.org/images/'
#     img_id = i.split('_', 1)
    
#     img_path = img_id[0] + '/' + img_id[1] + '_' + color + '.jpg'
#     img_url = base_url + img_path
#     return img_url


# def download(url, save_dir='../../data/input/moredata/train_source', image_size=(512, 512)):
#     img_name = '_'.join(url.split('/')[-2:]).split('.')[0] + '.jpg'

#     # r = requests.get(url, allow_redirects=True, stream=True)
#     # r.raw.decode_content = True

#     # # Use PIL to resize the image and to convert it to L
#     # # (8-bit pixels, black and white)
#     # im = Image.open(r.raw)
#     # im = im.resize(image_size, Image.LANCZOS).convert('L')
#     # im.save(os.path.join(save_dir, img_name), 'PNG')
#     # data = urlopen(url).read()
#     num = 5
#     try:
#         data = urlopen(url).read()
#     except Exception as e:
#         if num == 0:
#             return
#         else:
#             download(url)
#             num -= 1
#     with open(os.path.join(save_dir, img_name), 'wb') as f:
#         f.write(data)
#     print('1')


# def gevent_task(urls):
#     p = pool.Pool(100)
#     pid = os.getpid()
#     print('Task: %s start' % pid)

#     urls = urls.tolist()
#     tasks = [p.spawn(download, url) for url in urls]
#     print('Task Downloading...')
#     p.joinall(tasks)
#     print('Task: %s done' % pid)


# def main():
#     process_num = os.cpu_count()
#     df = pd.read_csv('../../data/input/moredata/HPAv18RBGY_wodpl.csv')
#     colors = ['red', 'green', 'blue', 'yellow']

#     for color in colors:
#         df[color] = df.Id.map(lambda x: make_url(x, color))
    
#     urls = pd.concat([df[color] for color in colors])
#     # gevent_task(urls)
#     # print(0)

#     len_urls = len(urls)
#     batch_size = int(np.ceil(len_urls / (process_num)))

#     url_list = []
#     for i in range(0, len_urls, batch_size):
#         url_list.append(urls.iloc[i:i+batch_size])
#     print(0)
#     pool = Pool(process_num)
#     print('Downloading...')
#     pool.map(gevent_task, url_list)
#     print('all done')


# if __name__ == '__main__':
#     main()



