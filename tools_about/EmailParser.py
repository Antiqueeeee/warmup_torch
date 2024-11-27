import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)
import email
from email.header import decode_header
from imapclient import IMAPClient
import io
import pandas as pd
import re
import json
from utils import recorder
from html import unescape
class Process:
    def __init__(self, username, password, target_email):
        self.username = username
        self.password = password
        self.target_email = target_email
        self.mail = IMAPClient("imap.163.com", port=993, ssl=True)
        self.mail.login(self.username, self.password)
        self.mail.id_({"name": "IMAPClient", "version": "2.1.0"})
        self.mail.select_folder("INBOX")
        self.processed_record = self.get_record()
        self.recorder = recorder(task_name="邮件处理工具")
    # 如果流程精细的话，从接到邮件开始，应该是一个任务开始执行，应该有完整的处理流程，中途报错也应该有特定的处理方式
    # BTW，任务执行完了应该还要将处理过的邮件数据更新到record当中
    def get_record(self):
        os.makedirs(os.path.join(project_path, "temporary"),exist_ok=True)
        _file = os.path.join(project_path, "temporary", "email_processed.json")
        if os.path.exists(_file) is False:
            with open(_file, "w" , encoding="utf-8") as f:
                json.dump(list(), f , ensure_ascii=False, indent=2)
            return list()
        
        else:
            with open(_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data


    def get_emails(self):
        criteria = ['ALL']
        messages = self.mail.search(criteria)
        emails = []
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

    def process_emails(self):
        emails = self.get_emails()
        links = list()
        downloaded = list()
        for (uid, email_message) in emails:
            for link in process.extract_hyperlinks(email_message):
                links.append((uid, link))

        for (uid, link) in links:
            file_name = os.path.join(project_path, "temporary", uid + ".zip")
            flag = self.attachment_download(link, file_name)
            if flag is True:
                downloaded.append(file_name)
        for down in downloaded:
            pass
    def attachment_download(self, url, file_name):
        import requests
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # 如果请求失败会抛出异常
            with open(file_name, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            self.recorder.info(f"邮箱工具下载附件成功：\n{url}")
            return True
        except Exception as e:
            self.recorder.error(f"邮箱工具下载附件失败:\n{url}\n错误原因：\n{str(e)}")
            return False
        
        


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

    target_email = "mapbioo@163.com"
    process = Process("mapgenedata@163.com", "FAifiQkpYHk3jrFr", target_email)
    emails = process.process_emails()
