import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, db

class FirebaseManager:
    def __init__(self):
        print("[FirebaseManager] Initializing...")
        self.db = None
        self.rtdb = None
        
        # Checking if already initialized (helpful in some environments like FastAPI reloads)
        if not firebase_admin._apps:
            cred_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
            if cred_json:
                try:
                    # Fix: Handle escaped newlines properly from single-line .env inputs
                    cred_dict = json.loads(cred_json.replace('\\n', '\n'))
                    cred = credentials.Certificate(cred_dict)
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': 'https://sita-3180c-default-rtdb.asia-southeast1.firebasedatabase.app'
                    })
                    self.db = firestore.client()
                    self.rtdb = db.reference()
                    print("[FirebaseManager] Firebase Connected Successfully (Dual-Write enabled).")
                except Exception as e:
                    print(f"[FirebaseManager] Firebase Init Error: {e}")
            else:
                print("[FirebaseManager] WARNING: FIREBASE_SERVICE_ACCOUNT_JSON not found. DB writes skipped.")
        else:
            self.db = firestore.client()
            self.rtdb = db.reference()

    def upload_vehicle_data(self, vehicle_id, class_name, plate_text, color):
        if not self.db or not self.rtdb:
            return
        
        # Structure for both databases
        data = {
            'id': str(vehicle_id),
            'type': class_name,
            'plate': plate_text,
            'color': color,
            'timestamp': firestore.SERVER_TIMESTAMP # Will become actual server time in Firestore
        }
        
        # 1. Firestore Write
        try:
            doc_ref = self.db.collection('vehicles').document(str(vehicle_id))
            doc_ref.set(data)
            print(f"[FirebaseManager] Firestore Uploaded ID: {vehicle_id}")
        except Exception as e:
            print(f"[FirebaseManager] Firestore Error: {e}")
            
        # 2. Realtime Database Write
        rtdb_data = data.copy()
        import datetime
        rtdb_data['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        try:
            # Update latest vehicle
            self.rtdb.child('live_traffic').child('latest').set(rtdb_data)
            
            # Increment total counter across transactions safely
            counter_ref = self.rtdb.child('live_traffic').child('total_count')
            current_count = counter_ref.get() or 0
            counter_ref.set(current_count + 1)
            print(f"[FirebaseManager] RTDB Updated. Total Count: {current_count + 1}")
        except Exception as e:
            print(f"[FirebaseManager] RTDB Error: {e}")
