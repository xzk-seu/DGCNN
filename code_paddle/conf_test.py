import configparser

cf = configparser.ConfigParser()
cf.read("dgcnn_paddle.conf")
t = cf.sections()
print(t)
cf.get("model", "")
