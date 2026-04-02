import numpy as np

def predict_future(model, last_values, days=7):
    temp = last_values.copy()
    preds = []

    for _ in range(days):
        x = np.array(temp[-5:]).reshape(1, -1)
        pred = model.predict(x)[0]
        preds.append(pred)
        temp.append(pred)

    return preds