import sqlite3

db = sqlite3.connect("trainingData.db")
db.close()

db = sqlite3.connect("file:trainingData.db?mode=rwc", uri=True)
cursor = db.cursor()
sql_command = """
    CREATE TABLE trainingExamples (
        id INT PRIMARY KEY,
        newDirection INT,
        aiSensor_0_0 INT,
        aiSensor_0_1 INT,
        aiSensor_0_2 INT,
        aiSensor_1_0 INT,
        aiSensor_1_1 INT,
        aiSensor_1_2 INT,
        aiSensor_2_0 INT,
        aiSensor_2_1 INT,
        aiSensor_2_2 INT,
        aiSensor_3_0 INT,
        aiSensor_3_1 INT,
        aiSensor_3_2 INT,
        direction INT
    )
"""  

# sql_command = "DELETE FROM trainingExamples WHERE aiSensor_0_0 = 0 or aiSensor_1_0 = 0 or aiSensor_2_0 = 0 or aiSensor_3_0 = 0"
cursor.execute(sql_command)
# rows = cursor.fetchall()
# for row in rows:
#     print(row[14])
db.commit()
db.close()