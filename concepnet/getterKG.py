import requests
import time
import networkx as nx
import matplotlib.pyplot as plt

def consultar_conceptnet(concepto, idioma="en"):
    url = f"http://api.conceptnet.io/c/{idioma}/{concepto}"
    try:
        response = requests.get(url).json()
    except Exception as e:
        print(f"Error consultando {concepto}: {e}")
        return []

    relaciones = []
    for edge in response.get("edges", []):
        start = edge.get("start", {})
        end = edge.get("end", {})

        if start.get("language") != idioma or end.get("language") != idioma:
            continue

        sujeto = start.get("label", "")
        predicado = edge.get("rel", {}).get("label", "")
        objeto = end.get("label", "")

        if sujeto and predicado and objeto:
            relaciones.append((sujeto, predicado, objeto))
    
    return relaciones


def construir_kg_multiple(seeds, profundidad=2, idioma="en"):
    visitados = set()
    kg = set()

    def expandir(concepto, nivel):
        if nivel > profundidad or concepto in visitados:
            return
        visitados.add(concepto)
        relaciones = consultar_conceptnet(concepto, idioma)
        for sujeto, predicado, objeto in relaciones:
            tripleta = (sujeto, predicado, objeto)
            kg.add(tripleta)
            expandir(objeto.lower().replace(" ", "_"), nivel + 1)
            time.sleep(0.1)

    for seed in seeds:
        print(f"🔍 Expandiendo: {seed}")
        expandir(seed.lower().replace(" ", "_"), 0)
    
    return list(kg)

def guardar_kg(kg, nombre_archivo="conceptnet_kg.spo"):
    with open(nombre_archivo, "w", encoding="utf-8") as f:
        for s, p, o in kg:
            f.write(f"{s}\t{p}\t{o}\n")
    print(f"\n✅ Grafo guardado en '{nombre_archivo}' con {len(kg)} relaciones.")



# 🔧 USO
if __name__ == "__main__":
    palabras = ['feeling', 'alive', 'life', 'mental', 'cry', 'depressive', 'anxiety', 'help',
 'tired', 'kill', 'feel', 'happy', 'crying', 'support', 'helping', 'helps',
 'depression', 'helpful', 'hopeless', 'depressed', 'worthless', 'kills',
 'die', 'pain', 'feels', 'sadness', 'feelings', 'alone', 'love', 'sad']


    profundidad = 2

    grafo = construir_kg_multiple(palabras, profundidad)
    guardar_kg(grafo,"conceptnet_telegram.spo")
