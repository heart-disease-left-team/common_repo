import flask
import numpy as np
from sklearn.linear_model import LogisticRegression


app = flask.Flask("mcnulty-app")

# X = np.linspace(1,1000,50).reshape(-1,1)
# Y = np.zeros(50,)
# Y[25:] = np.ones(25,)
# PREDICTOR = LogisticRegression().fit(X,Y)

@app.route("/states")
def showchart():
  with open("negbar_mod.html",'r') as vizfile:
    return vizfile.read()
  
@app.route("/")
def hello():
  return "It's alive!"
  
@app.route("/gabe")
def gabe():
  page = "<html>"
  page += "<body>"
  page += "<p> HI IAM A <strong>WALRUS</strong>. OMG. </p>"
  page += "</body>"
  page += "</body>"
  page += "</html>"
  return page
  
app.debug = True

app.run(host='0.0.0.0', port=80)