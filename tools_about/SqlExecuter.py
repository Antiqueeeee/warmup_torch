import os, sys
current_path = os.path.abspath(os.path.join(__file__, "../"))
project_path = os.path.abspath(os.path.join(current_path, "../"))
sys.path.append(project_path)

from utils import recorder
from urllib.parse import urlparse
import time
import pymysql
class sql_handler:
    def __init__(self) -> None:
        self.user_name = os.environ.get('MYSQL_USERNAME', "root")
        self.password = os.environ.get('MYSQL_PASSWORD', "There is no password in envionment.")
        self.recorder = recorder("数据库")
        self.dburl = urlparse(os.environ.get("MYSQL_URL", "mysql://localhost:3306/science"))


    def execute_sql(self, sql, max_execute_times=3):
        self.recorder.info(f"执行sql:\n{sql}")
        execute_times = 1
        # 执行sql重试
        while execute_times < max_execute_times:
            try:
                conn, cursor = self.open_database()
                cursor.execute(sql)
                break
            except Exception as e:
                execute_times += 1
                self.recorder.error(f"sql执行发生错误，详细信息：\n{str(e)}")
                execute_times += 1
                time.sleep(0.5)
                continue

        # 查询执行成功，则返回数据，失败则返回空
        if sql.strip().upper().startswith("SELECT"):
            if execute_times < max_execute_times:
                data = cursor.fetchall()
                return data
            else:
                return list()
        # 其他操作若执行成功，则提交，否则返回False
        else:
            if execute_times < max_execute_times:
                conn.commit()
                return True
            else:
                return False
            
    def open_database():
        url = urlparse(os.environ.get("MYSQL_URL", "mysql://localhost:3306/science"))
        conn= pymysql.connect(
            host = url.hostname
            ,user = os.environ.get('MYSQL_USERNAME', "root")
            ,passwd = os.environ.get('MYSQL_PASSWORD', "There is no password in envionment.")
            ,db = url.path[1:]
            ,port=url.port or 3306
            ,charset="utf8"
        )
        cursor = conn.cursor()
        cursor.execute("select version()")
        data = cursor.fetchone()
        print(data)
        return conn, cursor
