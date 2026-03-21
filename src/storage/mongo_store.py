# storage/mongo_store.py

from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

class MongoStore:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGO_URI"))
        self.db = self.client["omniraq"]
        self.collection = self.db["documents"]

    def insert_chunks(self, chunks):
        self.collection.insert_many(chunks)

    def keyword_search(self, query, top_k=5, session_id=None):
        filter = {"text": {"$regex": query, "$options": "i"}}
        if session_id:
            filter["session_id"] = session_id

        results = self.collection.find(filter).limit(top_k)

        return list(results)

    def delete_by_session(self, session_id):
        if not session_id:
            return
        res = self.collection.delete_many({"session_id": session_id})
        return res.deleted_count