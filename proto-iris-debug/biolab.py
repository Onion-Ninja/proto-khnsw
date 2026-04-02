from pymongo import MongoClient
from pymongo.collection import Collection
from lib import LoggerManager as LM 
from lib import FileManager as FM
from lib import MongoJSONDecoder as MJDec
from lib import MongoJSONEncoder as MJEnc



# Setup logging
from functools import lru_cache
from difflib import get_close_matches

# from time import sleep
# logger = LoggerManager.get_logger(name=Path(__file__).stem)
# fm=FM()
# lg = lm.get_logger(__name__,level="INFO")
# set log level to info
# self.lg.setLevel(logging.DEBUG)
from datetime import datetime, timezone

# Create the function
def _iso_date(date_string=None):
    if date_string is None:
        return datetime.now(timezone.utc)
    
    if date_string.endswith('Z'):
        date_string = date_string[:-1] + '+00:00'
    
    return datetime.fromisoformat(date_string)

# Assign to ISODate to match MongoDB shell syntax
ISODate = _iso_date
from pathlib import Path
import json

class IrisDB():
    """
    In: None
    Out: IrisDB Object that can be used to connect to a particular db
    param: 
        ds_id - name of the dataset collection
    """
    DB_IP = 'localhost'
    DB_NAME = 'iris_db'
    META_COLL_NAME = 'meta'
    DB_BASE_ = Path(f"~/datasets/{DB_NAME}/").expanduser()

    def __init__(
        self, 
        ds_id=None,
        db_ip=None, 
        mongo_db_name=None, 
        meta_coll_name=None
        # db_coll=None, # this could be used to shortcircuit the process?
        ) -> object:
        self.db_ip = self.DB_IP if db_ip is None else db_ip
        self.mongo_db_name = self.DB_NAME if mongo_db_name is None else mongo_db_name
        self.meta_coll_name = self.META_COLL_NAME if meta_coll_name is None else meta_coll_name
        # self.db_coll = DB_COLL if db_coll is None else db_coll
        ## get user:passwd from mongo_creds.txt file
        self.lg = LM.get_logger(name=__name__,level="INFO")
        self.fm = FM()
        user,passwd = self.fm.read_creds()
        self._mongo_admin_user = user
        self._mongo_admin_password = passwd
        self.closing = False
        if ds_id is not None:
            self.connect(ds_id)

        
        # self.fm.ensure_exists(self.DB_BASE_)
        # set it have the same properties as Collection class
        
    @property
    @lru_cache(maxsize=None) # Caches the result after the first call
    def mongo_client(self):
        """Lazily creates and returns the MongoClient instance."""
        if self.closing:
            delattr(self, 'mongo_client')
            return None
        self.lg.debug("MongoDB client is not initialized. Creating client...")
        mc = MongoClient(
            self.db_ip,
            username=self._mongo_admin_user,
            password=self._mongo_admin_password,
            authSource="admin"
        )
        self.lg.debug("MongoDB client created successfully.")
        return mc
    

    @property
    @lru_cache(maxsize=None) # Caches the result after the first call
    def mongo_db(self):
        """Lazily creates and returns the Database object using the client."""
        if self.closing:
            delattr(self, 'mongo_db')
            return None
        self.lg.debug("Establishing MongoDB database connection...")
        # This will automatically trigger the mongo_client property if needed
        conn = self.mongo_client[self.mongo_db_name]
        self.lg.debug("MongoDB connection established successfully.")
        return conn
    


    @property
    @lru_cache(maxsize=None) # Caches the result after the first call
    def avail_ds(self) -> set:
        """LazilB creates and returns the set of available {DB_NAME} from meta coll using the mongo client."""
        self.lg.debug("Connecting to meta db to find avail ds")
        # This will automatically trigger the mongo_client property if needed
        try:
            avail_ds = set(self.mongo_db.list_collection_names())
        except Exception as e:
            self.lg.error(f"Error fetching available datasets: {e}")
            raise e
        # avail_ds.add(self.meta_coll_name)
        # print(avail_ds)
        self.lg.debug(f"Fetched List of avail databases -> {avail_ds}")
        return avail_ds

    def get_avail_ds(self) -> set:
        """Get a set of available IRIS ds_id."""
        print(f"Avail Datasets: {self.avail_ds}")
        return self.avail_ds
    get_datasets = get_ds = list_ds = get_avail_ds = get_avail_ds
    
    # @property
    # @lru_cache(maxsize=None) # Caches the result after the first call
    # def coll(self):
    #     """Lazily creates and returns the collection object using the database."""
    #     if self.closing:
    #         delattr(self, 'coll')
    #         return None
    #     self.lg.debug("Establishing MongoDB collection connection...")
    #     # This will automatically trigger the mongo_db property if needed
    #     coll = self.mongo_db[self.ds_name]
    #     self.lg.debug("MongoDB collection connection established successfully.")
    #     return coll

    def find_ds(self, ds_id, avail_ds=None, acc=0.4, count=1) -> str|set:
        # 1. Create a mapping from lowercase name to original name.
        mapping = {db.lower(): db for db in (avail_ds or self.avail_ds)}
        # 2. Get the lowercase versions of all available DBs for matching.
        lower_avail_ds = list(mapping.keys())
        # 3. Perform the match on the lowercase versions.
        matches = get_close_matches(ds_id.lower(), lower_avail_ds, n=count, cutoff=acc)
        # 4. If a match is found, use the mapping to return the original name.
        if matches:
            if count > 1:
                res = {mapping[match] for match in matches}
            else:
                res = mapping[matches[0]]
            msg = f"Found matches for {ds_id}: {res}"
            print(msg)
            self.lg.debug(msg)
            return res
        return None

    # @lru_cache(maxsize=None) # Caches the result after the first call
    # def set_meta_primary(self):
    #     """Get the meta collection"""
    #     self.lg.debug("Accessing meta collection...")
    #     self.ds_id = self.meta_coll_name
    #     self.coll = self.meta_coll
    #     return self.meta_coll
    # meta_connect = set_meta_primary

    # def meta_connect(self):
    #     """Connect to the meta collection"""
    #     self.lg.info("Connecting to meta collection...")
    #     return self.meta_coll

    def connect(self, ds_id, acc=0.4) -> Collection:
        """Will connect to the database into a given collection
        It sets the self.ds_id attrib and self.coll
        """
        # if meta is tring to be connected then return the meta collection
        # if ds_id == self.meta_coll_name:
        #     self.ds_id = self.meta_coll_name
        #     # self.coll = self.get_meta_coll()
        #     self.lg.info(f"Connecting to {ds_id} Collection")
        #     return self.meta_coll
        
        avail_ds = self.avail_ds
        if (closest_match := self.find_ds(ds_id=ds_id, avail_ds=avail_ds, acc=acc)):
            self.ds_id = closest_match
            self.lg.info(f"Connecting to {self.ds_id} Collection")
            # self.ds_prefix=Path(DS_PREFIX)
            # self.ds_path=self.ds_prefix/self.ds_id
        else:
            self.lg.error(f"{ds_id} Collection Not found in available datasets. List: {avail_ds}")
            return None
        self.coll = self.mongo_db[self.ds_id]
        return self.coll
    get_coll = connect

    # def determine_coll(self,collection=None):
    #     if collection is None:
    #         if self.ds_id == self.meta_coll_name:
    #             collection = self.meta_coll
    #         else:
    #             collection = self.coll
    #     return collection
        
    def update(self, doc, key = None, coll=None):
        """Update a single document in the connected collection"""
        if '_id' in doc:
            key = '_id'
        if key is None:
            key = 'ds_id'
        if coll is None:
            coll = self.coll
        res=coll.update_one({key: doc[key]}, {'$set': doc}, upsert=False)
        self.lg.info(f"Updated document in {self.ds_id} collection.")
        return res
    
    def insert(self, docs, coll = None):
        if coll is None:
            coll = self.coll
        try:
            if isinstance(docs, dict):
                res = coll.insert_one(docs)
            elif isinstance(docs, list):
                # ignore if duplicate key error
                res = coll.insert_many(docs, ordered=False)
        except Exception as e:
            self.lg.error(f"Error inserting document into {self.ds_id} collection, error: {e}")
            res = None
        self.lg.info(f"Inserted document(s) into {self.ds_id} collection.")
        return res

    def find(self, query, proj=None, collection=None):
        """Get data from the connected collection"""
        if collection is None:
            if not hasattr(self, 'coll'):
                self.lg.error("No collection connected. Please call connect() first.")
                return None
            collection = self.coll

        return collection.find(query, proj)
    
    def find_one(self, query, proj=None, collection=None):
        """Get a single document from the connected collection"""
        if collection is None:
            if not hasattr(self, 'coll'):
                self.lg.error("No collection connected. Please call connect() first.")
                return None
            collection = self.coll

        return collection.find_one(query, proj)
    

        
    def get_num_eyes_per_person(self,tag='orig'):
        num_eyes_per_person = {}
        for person in self.find({'img_tags': tag}).distinct('person_id'):
            n = str(len(list(self.find({'person_id': person,'img_tags': tag}).distinct('eye_id'))))
            try:
                num_eyes_per_person[n].append(person)
            except KeyError:
                num_eyes_per_person[n] = [person]
        self.lg.debug(f"DS_ID: {self.ds_id} \n num_eyes_per_person: {num_eyes_per_person}")
        return num_eyes_per_person

    def get_num_samples_per_eye(self,tag='orig'):
        num_samples_per_eye = {}
        for eye in self.find({'img_tags': tag}).distinct('eye_id'):
            n = str(len(list(self.find({'eye_id': eye,'img_tags': tag,}))))
            try:
                num_samples_per_eye[n].append(eye)
            except KeyError:
                num_samples_per_eye[n] = [eye]
        self.lg.debug(f"DS_ID: {self.ds_id} \n num_samples_per_eye: {num_samples_per_eye}")
        return num_samples_per_eye


    def get_num_eyes_per_person_count(self,tag='orig'):
        num_eyes_per_person = self.get_num_eyes_per_person(tag)
        count = {k: len(v) for k, v in num_eyes_per_person.items()}
        self.lg.info(f"DS_ID: {self.ds_id} \n num_eyes_per_person_count: {count}")
        return count

    def get_num_samples_per_eye_count(self,tag='orig'):
        num_samples_per_eye = self.get_num_samples_per_eye(tag)
        count = {k: len(v) for k, v in num_samples_per_eye.items()}
        self.lg.info(f"DS_ID: {self.ds_id} \n num_samples_per_eye_count: {count}")
        return count

    def get_num_eyes(self,tag='orig'):
        count = len(self.find({'img_tags': tag}).distinct('eye_id'))
        self.lg.info(f"DS_ID: {self.ds_id} \n num_eyes: {count}")
        return count

    def get_num_people(self,tag='orig'):
        count = len(self.find({'img_tags': tag}).distinct('person_id'))
        self.lg.info(f"DS_ID: {self.ds_id} \n num_people: {count}")
        return count

    def get_num_images(self,tag='orig'):
        count=self.coll.count_documents({'img_tags': tag})
        self.lg.info(f"DS_ID: {self.ds_id} \n num_images: {count}")
        return count

    def get_session_count(self,tag='orig'):
        count = len(self.find({'img_tags': tag}).distinct('session_id'))
        self.lg.info(f"DS_ID: {self.ds_id} \n num_sessions: {count}")
        return count
    
    def get_stats(self, tag='orig', large=False):
        # img = self.find_one({}, {'_id': 0, })
        data = {
            # "info": "original images",
            "num_images": self.get_num_images(tag),
            "num_people": self.get_num_people(tag),
            "num_eyes": self.get_num_eyes(tag),
            "num_eyes_per_person_count": self.get_num_eyes_per_person_count(tag), 
            "num_samples_per_eye_count": self.get_num_samples_per_eye_count(tag),
            "num_sessions": self.get_session_count(tag),
        }
        if large:
            data.update({
            "num_eyes_per_person": self.get_num_eyes_per_person(tag),
            "num_samples_per_eye": self.get_num_samples_per_eye(tag),
        })
        return data





    ## feature to get a Mongo Collection by getitem on iris_db object
    def __getitem__(self, coll_name):
        """Get a MongoDB collection by name"""
        # if not hasattr(self, 'mongo_conn'):
            # self.lg.error("No MongoDB connection established.")
            # return None
        return self.get_coll(coll_name)

    # def find(self, query, proj=None, collection=None):
    #     """Get data from the connected collection"""
    #     if collection is None:
    #         if not hasattr(self, 'coll'):
    #             self.lg.error("No collection connected. Please call connect() first.")
    #             return None
    #         collection = self.coll

    #     return collection.find(query, proj)

    # def update_one(self, doc, key=None, collection=None):
    #     """Update a single document in the connected collection"""
    #     if not hasattr(self, 'coll'):
    #         self.lg.error("No collection connected. Please call connect() first.")
    #         return None
    #     if collection is None:
    #         collection = self.coll
    #     if key is None:
    #         if '_id' in doc:
    #             key = '_id'
    #         elif self.ds_name=='meta' and 'ds_id' in doc:
    #             key = 'ds_id'
    #         else:
    #             raise ValueError("No valid key found for document.")
    #     if key in doc:
    #         collection.update_one({key: doc[key]}, {'$set': doc}, upsert=True)
    #     else:
    #         raise ValueError(f"Key '{key}' not found in document.")
    #     self.lg.info(f"Updated document in {self.ds_name} collection.")
    #     return True
    
    # def update_many(self, docs, key=None):
    #     """Update data in the connected collection"""
    #     if not hasattr(self, 'coll'):
    #         self.lg.error("No collection connected. Please call connect() first.")
    #         return None
    #     if isinstance(docs, dict):
    #         docs = [docs]
    #     for doc in docs:
    #         if key is None:
    #             if '_id' in doc:
    #                 key = '_id'
    #             elif self.ds_name=='meta' and 'ds_id' in doc:
    #                 key = 'ds_id'
    #             else:
    #                 raise ValueError("No valid key found for document.")
    #         if key in doc:
    #             self.coll.update_one({key: doc[key]}, {'$set': doc}, upsert=True)
    #         else:
    #             raise ValueError(f"Key '{key}' not found in document.")
    #     self.lg.info(f"Updated {len(docs)} documents in {self.ds_name} collection.")
    #     return True



    # def insert_one(self, doc):
    #     """Insert a single document into the connected collection"""
    #     if not hasattr(self, 'coll'):
    #         self.lg.error("No collection connected. Please call connect() first.")
    #         return None
    #     self.coll.insert_one(doc)
    #     self.lg.info(f"Inserted document into {self.ds_name} collection.")
    #     return True
    def export_to_jsonl(self, query={}, proj={}, dest_path_=None, coll=None):
        """Export the connected collection to a jsonl file"""
        if coll is None:
            coll = self.coll
        if dest_path_ is None:
            dest_path_ = self.DB_BASE_/self.ds_id/f"{self.ds_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.fm.ensure_exists(dest_path_.parent)
        # Custom JSON encoder to handle MongoDB-specific types
        with dest_path_.open('w', encoding='utf-8') as jsonl__:
            for doc in coll.find(query, proj):
                jsonl__.write(json.dumps(doc, cls=MJEnc) + '\n')
        self.lg.info(f"Exported collection {self.ds_id} to {dest_path_}")
        return dest_path_

    def import_from_jsonl(self, src_path_=None, coll=None):
        """Import a jsonl file into the connected collection"""
        if coll is None:
            coll = self.coll
        
        if src_path_ is None:
            src_base_ = self.DB_BASE_/self.ds_id
            # find latest file in src_base_ with .jsonl extension
            jsonl_files = list(src_base_.glob("*.jsonl"))
            if not jsonl_files:
                self.lg.error(f"No JSONL files found in {src_base_}.")
                return None
            # Sort files by modification time and take the latest
            src_path_ = max(jsonl_files, key=lambda f: f.stat().st_mtime)

        inserted_ids = []
        with Path(src_path_).open('r', encoding='utf-8') as jsonl__:
            for line in jsonl__:
                doc = json.loads(line, cls=MJDec)
                try:
                    res = coll.insert_one(doc)
                    inserted_ids.append(res.inserted_id)
                except Exception as e:
                    self.lg.error(f"Error inserting document: {e}")
        self.lg.info(f"Imported {len(inserted_ids)} documents into collection {self.ds_id} from {src_path_}")
        return inserted_ids
    
    def __enter__(self):
        """Called when entering the 'with' statement."""
        self.lg.debug("Entering context...")
        return self # Return the instance to be used in the 'with' block
    
    def __exit__(self,*args):
        """Called when exiting the 'with' statement."""
        # This method is always called, ensuring the connection is closed.
        self.close()
        # self.lg.debug("MongoClient connection closed.")

    def close(self):
        """Explicitly close the mongo client connection"""  
        if self.closing:
            return
        try:
            self.closing = True
            if hasattr(self, 'mongo_client'):
                self.lg.debug("checking for mongoclient")
                self.mongo_client.close()
                self.lg.info("MongoDB client connection closed successfully.")
            else:
                self.lg.debug("MongoDB client was not initialized; no connection to close.")
        except Exception as e:
            self.lg.error(f"Error closing MongoDB client connection: {e}")
    
    def __del__(self):
        try:
            if self.closing:
                return
            self.close()
            self.lg.debug("MongoDB connection closed successfully.")
        except Exception as e:
            self.lg.error(str(e))


class IrisMeta(IrisDB):
    """
    The obj of this class is to handle metadata operations for the IrisDB.
    It will have methods to validate, update, and retrieve metadata.
    It will also have methods to handle versioning of metadata.
    """
    META_COLL_NAME = 'meta'
    def __init__(self, meta_coll_name=None):
        self.meta_coll_name = self.META_COLL_NAME if meta_coll_name is None else meta_coll_name
        # self.avail_ds = super().avail_ds + {self.meta_coll_name}
        super().__init__(ds_id=self.meta_coll_name, meta_coll_name=self.meta_coll_name)
        # self.connect(self.meta_coll_name)

    # @property
    # @lru_cache(maxsize=None) # Caches the result after the first call
    # def meta_coll(self):
    #     """Lazily creates and returns the Meta collection using the client."""
    #     coll = self.db.mongo_db[self.meta_coll_name]
    #     self.lg.debug("Returned meta collection.")
    #     return coll

    def get_metadata(self, ds_id: str, proj=None) -> dict:
        """Retrieve metadata for a given dataset ID."""
        if not self.find_ds(ds_id):
            self.lg.error(f"Dataset ID '{ds_id}' not found in available datasets.")
            return None
        if proj is None:
            proj = {"_id": 0}  # Exclude MongoDB internal ID by default
        metadata = self.find_one({"ds_id": ds_id}, proj=proj)
        if metadata:
            self.lg.info(f"Metadata retrieved for dataset ID '{ds_id}'.")
        else:
            self.lg.error(f"No metadata found for dataset ID '{ds_id}'.")
        return metadata
    
    def update_metadata(self, metadata: dict) -> bool:
        """Update metadata for a given dataset ID."""
        if "ds_id" not in metadata:
            self.lg.error("Metadata must contain 'ds_id' field for update.")
            return False
        if not self.find_ds(metadata["ds_id"]):
            self.lg.error(f"Dataset ID '{metadata['ds_id']}' not found in available datasets.")
            return False
        try:
            self.update(metadata)
            self.lg.info(f"Metadata updated for dataset ID '{metadata['ds_id']}'.")
            return True
        except Exception as e:
            self.lg.error(f"Error updating metadata for dataset ID '{metadata['ds_id']}': {e}")
            return False

    def insert_metadata(self, metadata: dict) -> bool:
        """Insert new metadata for a dataset."""
        if "ds_id" not in metadata:
            self.lg.error("Metadata must contain 'ds_id' field for insertion.")
            return False
        if self.find_ds(metadata["ds_id"]):
            self.lg.error(f"Dataset ID '{metadata['ds_id']}' already exists in available datasets.")
            return False
        try:
            self.insert(metadata)
            self.lg.info(f"Metadata inserted for dataset ID '{metadata['ds_id']}'.")
            return True
        except Exception as e:
            self.lg.error(f"Error inserting metadata for dataset ID '{metadata['ds_id']}': {e}")
            return False

    def list_datasets(self) -> list:
        """List all dataset IDs available in the metadata collection."""
        datasets = set(self.avail_ds - {self.meta_coll_name})
        self.lg.info(f"Available datasets: {datasets}")
        return datasets
    get_datasets = get_ds = list_ds = get_avail_ds = list_datasets
    
    def delete_metadata(self, ds_id: str) -> bool:
        """Delete metadata for a given dataset ID."""
        if not self.find_ds(ds_id):
            self.lg.error(f"Dataset ID '{ds_id}' not found in available datasets.")
            return False
        try:
            result = self.coll.delete_one({"ds_id": ds_id})
            if result.deleted_count > 0:
                self.lg.info(f"Metadata deleted for dataset ID '{ds_id}'.")
                return True
            else:
                self.lg.error(f"No metadata found to delete for dataset ID '{ds_id}'.")
                return False
        except Exception as e:
            self.lg.error(f"Error deleting metadata for dataset ID '{ds_id}': {e}")
            return False
    
    # @staticmethod
    # def get_def_img_tag_data():
    #     tag_data = {
    #         "info": "",
    #         # "num_images": 0,
    #         # "num_people": 0,
    #         # "num_eyes": 0,
    #         # "num_eyes_per_person": {},
    #         # "num_eyes_per_person_count": {},
    #         # "num_samples_per_eye": {},
    #         # "num_samples_per_eye_count": {},
    #         # "num_sessions": [],
    #         # "img_specs": {},
    #         "orig_base_path": ""
    #     }
    #     return tag_data

    # def insert_new_img_tag(self,tag:str,tag_data:dict, ds_id:str=None):
    #     """Insert a new image tag into the metadata for a given dataset ID. 
    #     Eg: 'fv_tags'
    #     """
    #     if ds_id is None:
    #         ds_id = self.ds_id
    #     if not self.find_ds(ds_id):
    #         self.lg.error(f"Dataset ID '{ds_id}' not found in available datasets.")
    #         return False
    #     metadata = self.get_metadata(ds_id)
    #     if 'fv_tags' not in metadata:
    #         metadata['fv_tags'] = []
    #     if tag in metadata['fv_tags']:
    #         self.lg.error(f"Tag '{tag}' already exists in dataset ID '{ds_id}'.")
    #         return False
    #     metadata['fv_tags'].append(tag)
    #     metadata[tag] = tag_data
    #     try:
    #         self.update_metadata(metadata)
    #         self.lg.info(f"Tag '{tag}' inserted into metadata for dataset ID '{ds_id}'.")
    #         return True
    #     except Exception as e:
    #         self.lg.error(f"Error inserting tag '{tag}' into metadata for dataset ID '{ds_id}': {e}")
    #         return False

    def insert_new_img_tag(self,tag:str,tag_data:dict,ds_id:str=None):
        """Insert a new tag into the metadata for a given dataset ID. 
        Eg: 'norm_def'
        """
        if ds_id is None:
            ds_id = tag_data.get('ds_id', None)
            if ds_id is None:
                self.lg.error("Dataset ID must be provided either as a parameter or within tag_data.")
                return False
        if not self.find_ds(ds_id):
            self.lg.error(f"Dataset ID '{ds_id}' not found in available datasets.")
            return False
        metadata = self.get_metadata(ds_id, proj={"ds_id":1, "img_tags": 1, tag: 1, "_id": 0})
        if 'img_tags' not in metadata:
            metadata['img_tags'] = []
        if tag in metadata['img_tags']:
            self.lg.debug(f"Tag '{tag}' already exists in dataset ID '{ds_id}'.")
            # metadata['img_tags']=
            # return False
        else:
            metadata['img_tags'].append(tag)
        if tag in metadata:
            self.lg.debug(f"Tag '{tag}' already exists in dataset ID '{ds_id}', updating its data.")
            # return False
            metadata[tag].update(tag_data[tag])
        else:
            metadata[tag] = tag_data[tag]
        try:
            self.update_metadata(metadata)
            self.lg.info(f"Tag '{tag}' inserted into metadata for dataset ID '{ds_id}'.")
            return True
        except Exception as e:
            self.lg.error(f"Error inserting tag '{tag}' into metadata for dataset ID '{ds_id}': {e}")
            return False
         
class Iris:
    """
    The obj of this class is an Iris image and its associated metadata.
    The IrisDB class will return a list of Iris objects.
    Iris objects will have a feature to lazily load the image using opencv from the path only when called via iris_obj.load_image().
    Iris objects also have metadata attributes that can be accessed directly.
    Iris objects are created by the IrisDB class when querying the database.
    Iris objects can also be displayed to view contents.
    """
    def __init__(self):
        self.image_path = None
        self.md = {}


# Example usage:
if __name__ == "__main__":
    iris_db = IrisDB()
    iris_meta = IrisMeta()
    iris_meta.list_datasets()