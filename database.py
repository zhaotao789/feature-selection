import mysql.connector

class database():

    def __init__(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="root",
            database="serch"
        )
        self.mycursor = self.mydb.cursor()

    def create_database(self):
        self.mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="root"
        )
        mycursor = self.mydb.cursor()
        mycursor.execute("CREATE DATABASE serch")
        self.mydb.commit()

    def create_root_table(self):
        self.mycursor.execute("CREATE TABLE root_search_1(root_id int, root_acc FLOAT)")
        self.mydb.commit()

    def create_seed_table(self):
        self.mycursor.execute("CREATE TABLE seed_search_1(seed_id int, root_acc FLOAT,locate_id VARCHAR(65531))")
        self.mydb.commit()

    def create_search_table(self):
        self.mycursor.execute("CREATE TABLE search_search_1(id int, search_id VARCHAR(255), root_acc VARCHAR(255))")
        self.mydb.commit()

    def create_local_table(self):
        self.mycursor.execute(
            "CREATE TABLE local_search_1(id int, search_id VARCHAR(255), root_acc VARCHAR(255))")
        self.mydb.commit()

    def drop_table(self,x):
        self.mycursor.execute("DROP TABLE %s"%(x))
        self.mydb.commit()

    #第一次插入
    def insert(self,x,y):
        sql = "INSERT INTO root_search_1(root_id, root_acc) VALUES (%s, %s)"
        for i in range(len(y)):   #x=[[0],[1]]
            x1=int(x[i])
            y1=float('%.5f' % y[i])    #float('%.2f' % a)
            val = (x1, y1)
            self.mycursor.execute(sql, val)
            self.mydb.commit()
        return True

    def insert1(self,sql):
        #sql = "INSERT INTO seed_search(seed_id, root_acc) VALUES (%s, %s)"%(x,y)

        self.mycursor.execute(sql)
        self.mydb.commit()
        return True

    def insert2(self,z,x,y):

        sql="INSERT INTO search_search_1(id, search_id, root_acc) VALUES (%s, %s, %s)"
        val=(z,str(x),y)
        self.mycursor.execute(sql,val)
        self.mydb.commit()
        return True

    def update2(self,z,x,y):

        sql="UPDATE search_search_1 SET search_id=%s, root_acc=%s WHERE id=%s"
        val=(x,y,z)
        self.mycursor.execute(sql,val)
        self.mydb.commit()
        return True



    def UPDATE(self,x,y,z):
        sql = "UPDATE seed_search_1 SET locate_id=%s,root_acc=%s WHERE seed_id=%s"
        val=(x,y,z)
        self.mycursor.execute(sql,val)
        self.mydb.commit()
        return True



    def delete(self,sql):
        #sql = "DELETE FROM root_search WHERE root_id=%s" %(x)
        self.mycursor.execute(sql)
        self.mydb.commit()

    def select(self,sql):
          #SELECT root_id,root_acc FROM root_search
        self.mycursor.execute(sql)
        result = self.mycursor.fetchall()
        self.mydb.commit()
        return result




if __name__ == '__main__':
    database().drop_table('seed_search_1')
    database().drop_table('root_search_1')
    #database().drop_table('local_search_1')
    database().drop_table('search_search_1')
    database().create_root_table()
    database().create_seed_table()
    #database().create_local_table()
    database().create_search_table()
    #database().drop_table('seed_search')


