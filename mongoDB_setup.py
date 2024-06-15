import pymongo

MONGO_URI = "mongodb+srv://prajjwalguhe:10June2000@cluster0.ub6xf4f.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
DB_NAME = "thyroidDiseaseDB"

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

db.create_collection("patients")
print("Collection 'patients' created in database 'thyroidDiseaseDB'")
