import os
import urllib.request
import zipfile

def read_data_nmt():
    """下载并读取“英–法”数据集，返回文本内容字符串。"""
    # 1. 指定下载目录
    download_dir = './data/fra-eng/'
    os.makedirs(download_dir, exist_ok=True)

    # 2. 指定 zip 文件名和远程 URL
    zip_filename = os.path.join(download_dir, 'fra-eng.zip')
    url = 'https://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    # （如果上面 URL 无法使用，可尝试：https://resources.d2l.ai/data/fra-eng.zip）

    # 3. 下载 zip 文件（如果本地已存在就跳过）
    if not os.path.exists(zip_filename):
        print(f'Downloading {url} to {zip_filename} ...')
        urllib.request.urlretrieve(url, zip_filename)
        print('Download finished.')

    # 4. 解压到下载目录，下次运行就会跳过重新解压
    with zipfile.ZipFile(zip_filename, 'r') as zf:
        # 这里会解压出一个子文件夹 fra-eng，里面有 fra.txt
        zf.extractall(download_dir)

    # 5. 构造数据文件的完整路径并读取
    data_file = os.path.join(download_dir, 'fra-eng', 'fra.txt')
    with open(data_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    return raw_text

# 测试
if __name__ == '__main__':
    text = read_data_nmt()
    print('前500字符：\n', text[:500])
