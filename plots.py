import ast
import matplotlib.pyplot as plt

with open("results.txt", "r", encoding="utf-8") as arquivo:
    content = arquivo.read()

content = content.replace("INFO : ", "")

data = ast.literal_eval(content)

accuracy = [float(valor["accuracy"]) for valor in data.values()]
losses = [float(valor["loss"]) for valor in data.values()]

print("Acurácias:")
print(accuracy[:10])

print("\nLoss:")
print(losses[:10])

plt.figure(figsize=(10, 5))
plt.plot(accuracy)

plt.xlabel("Índice")
plt.ylabel("Acurácia")
plt.title("Acurácias do Modelo por parte do Servidor")
plt.grid(True)

plt.show()


plt.figure(figsize=(10, 5))
plt.plot(losses)

plt.xlabel("Índice")
plt.ylabel("Loss")
plt.title("Loss do Modelo por parte do Servidor")
plt.grid(True)

plt.show()