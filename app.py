from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

# Historical Data
data = pd.DataFrame({
    'Month':[1,2,3,4,5,6,7,8,9,10,11,12],
    'Sales':[1200,1350,1280,1420,1510,1650,
             1720,1800,1900,2050,2150,2250]
})

X = data[['Month']]
y = data['Sales']

# Train Model
model = LinearRegression()
model.fit(X,y)

# Create Graph
os.makedirs("static", exist_ok=True)

plt.figure(figsize=(6,4))
plt.scatter(X,y)

plt.plot(
    X,
    model.predict(X)
)

plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Sales Trend Prediction")

plt.savefig("static/graph.png")
plt.close()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    month = int(request.form['month'])

    result = model.predict([[month]])

    prediction = f"Predicted Sales : {result[0]:.2f}"

    return render_template(
        "result.html",
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)