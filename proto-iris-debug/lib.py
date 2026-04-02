#%% lib.py
# from json import load, dumps
from pathlib import Path
## Logger setup
import logging
pymongo_logger = logging.getLogger('pymongo')
# Set its level to WARNING to silence INFO and DEBUG messages
pymongo_logger.setLevel(logging.ERROR)


class LoggerManager:
    """Singleton Logger Manager for consistent logging across the app."""
    _logger = None

    @classmethod
    def get_logger(cls,  name=__name__, level="INFO", log_dir="logs/",):
        if cls._logger is None:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / f"{name}.log"
            logging.basicConfig(
                level=logging.DEBUG,
                # also log the calling function name
                format='%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s',
                # datefmt='%Y-%m-%d %H:%M:%S',
                handlers=[
                    logging.FileHandler(log_file, encoding="utf-8"),
                    logging.StreamHandler()
                ]
            )
            # set logging level
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)
            cls._logger = logging.getLogger(name)
            cls._logger.setLevel(level)
        return cls._logger

import shutil
class FileManager:
    """Singleton File Manager for consistent file operations."""
    _instance = None
    lg = LoggerManager.get_logger(name=__name__,level="ERROR")
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FileManager, cls).__new__(cls)
        return cls._instance

    # @staticmethod
    # def load_json_file(filename):
    #     """Load and return data from a JSON file."""
    #     with open(filename, 'r') as file:
    #         return load(file)

    # @staticmethod
    # def write_json_file(filename, data):
    #     """Write data to a JSON file."""
    #     with open(filename, 'w') as file:
    #         file.write(dumps(data, indent=4, ensure_ascii=False))
    
    # @staticmethod
    def read_creds(self,filename="mongo_creds.txt"):
        """Read MongoDB credentials from a file using pathlib"""
        
        if not Path(filename).exists():
            _ERROR_MSG = f"Credentials file {filename} does not exist.Please create a mongo_creds.txt with user:passwd"
            self.lg.error(_ERROR_MSG)
            raise FileNotFoundError(_ERROR_MSG)
        user, passwd = Path(filename).read_text().strip().split(":")
        self.lg.debug("MongoDB credentials loaded successfully.")
        return user, passwd
    
    # @staticmethod
    def ensure_exists(self,path):
        # check if path is Path() type if not make it
        if not isinstance(path, Path):
            path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        self.lg.debug(f"Ensured directory exists: {path}")
    
    # @staticmethod
    def copy_file(self, src, dest, overwrite=False):
        if not isinstance(dest, Path):
            dest = Path(dest)
        if not overwrite and dest.exists():
            self.lg.debug(f"File {dest} already exists. Skipping copy.")
            return
        self.ensure_exists(dest.parent)
        shutil.copy2(src, dest)
        self.lg.debug(f"Copied file from {src} to {dest}")



#%% Testing
# lg.WARNING
# import LoggerManager as lm
# logging.WARNING
# %%
import json
from bson import ObjectId
from datetime import datetime

class MongoJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
    
class MongoJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    # Attempt to convert _id key to ObjectId
                    if key == "_id":
                        obj[key] = ObjectId(value)
                except Exception:
                    pass
                try:
                    # Attempt to convert string to datetime
                    if 'T' in value and (len(value) >= 19):  # Basic check for ISO format
                        obj[key] = datetime.fromisoformat(value)
                except Exception:
                    pass
        return obj

