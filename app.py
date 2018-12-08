from flask import Flask, request, render_template
import json
from algorithm import inference

app = Flask(__name__)


@app.route('/')
def hello_world():
    # return re
    return render_template("index.html")


@app.route('/detect/', methods=['POST'])
def detect():
    data = request.get_data()
    json_data = json.loads(data.decode("utf-8"))
    base64 = json_data.get("base64")
    out_base64 = inference.detect_car(base64)


    poses = []
    poses.append([1.0, 2.0, 30, 4.0])
    # print "base64 : %s" % base64
    return json.dumps({"boxes":poses, "img":out_base64})


if __name__ == '__main__':
    app.run()
