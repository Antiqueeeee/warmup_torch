import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)
import email
from imapclient import IMAPClient
import zipfile
import pandas as pd
import requests
import re
import json
from utils import recorder
from html import unescape

tasks = ["药物有效性"]


class Process:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.mail = IMAPClient("imap.163.com", port=993, ssl=True)
        self.mail.login(self.username, self.password)
        self.mail.id_({"name": "IMAPClient", "version": "2.1.0"})
        self.mail.select_folder("INBOX")
        self.processed_record = self.get_record()
        self.recorder = recorder(task_name="邮件处理工具")
    # 如果流程精细的话，从接到邮件开始，应该是一个任务开始执行，应该有完整的处理流程，中途报错也应该有特定的处理方式

    def get_record(self):
        os.makedirs(os.path.join(project_path, "temporary"),exist_ok=True)
        _file = os.path.join(project_path, "records_about", "email_processed.json")
        if os.path.exists(_file) is False:
            with open(_file, "w" , encoding="utf-8") as f:
                json.dump(list(), f , ensure_ascii=False, indent=2)
            return list()
        
        else:
            with open(_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data

    def update_record(self):
        _file = os.path.join(project_path, "records_about", "email_processed.json")
        with open(_file, "w" , encoding="utf-8") as f:
            json.dump(self.processed_record, f , ensure_ascii=False, indent=2)

    def get_emails(self):
        criteria = ['ALL']
        messages = self.mail.search(criteria)
        emails = []
        self.processed_record = self.get_record()
        for uid in messages:
            response = self.mail.fetch(uid, ['RFC822', 'INTERNALDATE'])
            raw_message = response[uid][b'RFC822']
            email_message = email.message_from_bytes(raw_message)
            receive_date = response[uid][b'INTERNALDATE'].strftime("%Y%m%d")
            unique_id = f"{uid}-{receive_date}"
            if unique_id not in self.processed_record:
                emails.append((unique_id, email_message))
        return emails

    def extract_hyperlinks(self, email_message):
        hyperlinks = []
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/html":
                    html_content = part.get_payload(decode=True).decode(part.get_content_charset())
                    # 先提取链接，然后对每个链接进行HTML实体解码
                    links = re.findall(r'href=[\'"]?([^\'" >]+)', html_content)
                    hyperlinks.extend([unescape(link) for link in links])
        else:
            if email_message.get_content_type() == "text/html":
                html_content = email_message.get_payload(decode=True).decode(email_message.get_content_charset())
                links = re.findall(r'href=[\'"]?([^\'" >]+)', html_content)
                hyperlinks.extend([unescape(link) for link in links])
        
        hyperlinks = [i for i in hyperlinks if i.endswith(".zip") and "mapbioo" in i]
        if len(hyperlinks) > 0:
            hyperlinks = [hyperlinks[-1]]
        return hyperlinks
    
    def attachment_download(self, url, file_name):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 如果请求失败会抛出异常
            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            return True, "Success"
        except Exception as e:
            return False, str(e)
        
    def process_emails(self):
        emails = self.get_emails()
        links = list()
        downloaded = list()
        # 找到所有未登记过的迈浦附件
        for (uid, email_message) in emails:
            for link in self.extract_hyperlinks(email_message):
                self.recorder.info(f"发现未处理过的邮件ID:\t{uid}")
                links.append((uid, link))
        # 下载附件
        for (uid, link) in links:
            self.recorder.info(f"开始下载邮件中附件:\t{uid}")
            file_name = os.path.join(project_path, "temporary", uid + ".zip")
            flag, msg = self.attachment_download(link, file_name)
            if flag is True:
                downloaded.append(file_name)
                self.recorder.info(f"邮箱ID:{uid}-下载附件成功")
            else:
                self.recorder.error(f"邮箱ID:{uid}-下载附件失败，错误原因：\n{msg}")

        # 解压文件
        unpressed = list()
        for down in downloaded:
            file_name, file_fix = os.path.splitext(os.path.basename(down))
            unpress_path = os.path.join(os.path.dirname(down), file_name)
            os.makedirs(unpress_path, exist_ok=True)
            with zipfile.ZipFile(down, "r", metadata_encoding="gbk") as f:
                for file in f.namelist():
                    f.extract(file, unpress_path)
            unpressed.append(unpress_path)
            

        # 找到需要的数据, 按邮件ID和日期重命名
        data_required = list()
        for _path in unpressed:
            _tag = os.path.basename(_path)
            for root, dirs, files in os.walk(_path):
                for file in files:
                    file_name, file_fix = os.path.splitext(file)
                    _path = os.path.join(root, file)
                    new_name = os.path.join(root, _tag + file_fix)
                    if ".xls" in file  or ".xlsx" in file:
                        try: # 可能上一次任务正在执行，还未登记时，会报错
                            os.rename(_path, new_name)
                        except:
                            pass
                        data_required.append(new_name)
                        
        # 读取数据，重新保存到相应位置
        to_be_processed = list()
        for data in data_required:
            frame = pd.read_excel(data, sheet_name="Marker")
            file_name, file_fix = os.path.splitext(os.path.basename(data))
            for task in tasks:
                _path = os.path.join(project_path, 'tasks', '药物有效性', 'datasets', file_name + '.xlsx')
                frame.to_excel(_path, index=False)
                to_be_processed.append({
                        "task" : task
                        ,"instruction" : "模型推理"
                        ,"selected_model" : f"{task}_逻辑回归_20241126_1.pkl"
                        ,"inference_data" : os.path.basename(_path)
                    })
        return to_be_processed
        # # 唤起请求
        # for processing in to_be_processed:
        #     response = requests.post("http://127.0.0.1:8000/runCommand",json=processing)
        # 轮询预测结果，更新处理过的数据记录
    
        # 将process_email设置为定时任务


if __name__ == '__main__':
    # # process = Process("981722694@qq.com", "mnhbdpsarezcbfce", "com")
    # process = Process("mapgenedata@163.com", "FAifiQkpYHk3jrFr", "com")
    # emails = process.get_emails()
    # for email_message in emails:
    #     res = process.process_emails(email_message)
    #     if len(res) == 0:
    #         continue
    #     for sheet_name, data in res[0][1].items():
    #         print(sheet_name)
    #         print(data)
    #         print()

    # process = Process("mapgenedata@163.com", "FAifiQkpYHk3jrFr")
    # emails = process.process_emails()
    params = {
        "task" : "药物有效性"
        ,"instruction" : "模型推理"
        ,"selected_model" : f"药物有效性_逻辑回归_20241126_1.pkl"
        ,"inference_data" : os.path.basename('20241126_检测结果.xlsx')
        }
    response = requests.post("http://127.0.0.1:8000/runCommand", json=params).json().get("tasks_id", str())
    response
    print(response)