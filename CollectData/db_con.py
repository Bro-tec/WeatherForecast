import sqlite3
import pandas as pd
import numpy as np

def CreateTable(sqlPath,sql):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    mycursor.execute(sql)
    mydb.commit()
    mycursor.close()
    mydb.close()

# Hiermit werden die Farbschemen gespeichert
def write(sqlPath, dbName, df):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    df.to_sql(name=dbName, con=mydb, if_exists='replace',index = False, chunksize = 1000)
    mycursor.close()
    mydb.close()

# Wir brauchen die ID indem wir den Pfad schon haben 
def Read(sqlPath,dbName):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT * FROM {dbName}'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    return result
    # if len(result["iID"]) > 0:
    #     return result
    # else:
    #     print("klappt nicht")
    #     return None

def ReadyByCity(sqlPath,dbName,city):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT * FROM {dbName} WHERE Cityname="{city}"'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    return result

def getPosition(sqlPath, cityName):
    mydb = sqlite3.connect(sqlPath)
    mycursor = mydb.cursor()
    sql_query = f'SELECT * FROM cities where cityname="{cityName}"'
    result = pd.read_sql(sql_query, con=mydb)
    mycursor.close()
    mydb.close()
    return result
