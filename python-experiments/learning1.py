import psycopg2

conn =  psycopg2.connect("host=127.0.0.1 port=5432 dbname=shreeerp user=sa password=moraya")

cur = conn.cursor()

#cur.execute("select * from core.docs;")

#print(cur.rowcount)

#print(cur.fetchall())
#conn.commit()
# conn.close() 

cur.execute("select * from core.docs limit 2")
a = cur.fetchall()

print(a[0])
print(a[1])
print(type(a))


if a[1][0] == 1:
    print("okay")
else:
    print("not okay")

tuple1 = tuple(a)
print(tuple1)

tuple2 = [a[0][0], a[1][0]]
print(tuple2)

tuple2 = tuple(tuple2)
print(tuple2)

tuple2 = tuple2 + (3,5)
print(tuple2)
