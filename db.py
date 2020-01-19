import sqlite3

db = sqlite3.connect("trainingData.db")
db.close()

db = sqlite3.connect("file:trainingData.db?mode=rwc", uri=True)
cursor = db.cursor()
sql_command = """
    INSERT INTO testData (aiSensor_0_0, aiSensor_0_1, aiSensor_0_2, aiSensor_1_0, aiSensor_1_1, aiSensor_1_2, aiSensor_2_0, aiSensor_2_1, aiSensor_2_2, aiSensor_3_0, aiSensor_3_1, aiSensor_3_2, direction)
    VALUES (2,3,2,4,3,5,2,5,4,6,4,3,3);
"""  
cursor.execute(sql_command)
db.commit()
db.close()