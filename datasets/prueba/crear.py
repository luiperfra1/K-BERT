from datasets import load_dataset

# Cargar el conjunto de datos CoNLL-2003
dataset = load_dataset("conll2003")

# Acceder a las divisiones del conjunto de datos
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]

# Mostrar un ejemplo del conjunto de entrenamiento
print(train_data[0])
