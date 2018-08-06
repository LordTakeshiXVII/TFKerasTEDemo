import numpy as np
from tensorflow.keras.models import load_model

code_to_name = {0: "setosa", 1: "versicolor", 2: "virginica"}

model = load_model(".\model\iris.h5")

x = np.array([[5.9, 3, 4.2, 1.5]])
y = model.predict([x])
print(y)

code = np.argmax(y)
print(code)
print(code_to_name[code])