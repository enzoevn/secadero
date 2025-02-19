import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class Lote:
    def __init__(self, lote_id, data):
        self.lote_id = lote_id
        self.data = data[lote_id]
        self.producto = self.data["Producto"]
        self.fecha_produccion = self.data["Fecha de producción"]
        self.pesajes = self.data["Pesajes"]
        self.df = self._crear_dataframe()

    def _crear_dataframe(self):
        rows = []
        for pesaje in self.pesajes:
            fecha = pesaje["Fecha"]
            for pieza, peso in pesaje["Pesos"].items():
                peso_valor = float(peso.replace(" kg", "").replace(",", "."))
                rows.append({"Fecha": fecha, "Pieza": pieza, "Peso": peso_valor})
        df = pd.DataFrame(rows)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df

    def obtener_pesos(self):
        return self.df

    def obtener_fechas(self):
        return self.df['Fecha'].unique()

    def calcular_promedios(self):
        return self.df.groupby('Fecha')['Peso'].mean().reset_index()

    def graficar_evolucion_peso(self):
        plt.figure(figsize=(10, 6))
        for pieza in self.df['Pieza'].unique():
            subset = self.df[self.df['Pieza'] == pieza]
            plt.plot(subset['Fecha'], subset['Peso'], marker='o', label=pieza)
        plt.xlabel('Fecha')
        plt.ylabel('Peso (kg)')
        plt.title(f'Evolución del peso por fecha - Lote {self.lote_id}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def graficar_promedios(self):
        promedios = self.calcular_promedios()
        plt.figure(figsize=(10, 6))
        plt.plot(promedios['Fecha'], promedios['Peso'], marker='o', label='Promedio')
        plt.xlabel('Fecha')
        plt.ylabel('Peso Promedio (kg)')
        plt.title(f'Peso promedio por fecha - Lote {self.lote_id}')
        plt.legend()
        plt.grid(True)
        plt.show()

class PrediccionLote(Lote):
    def __init__(self, lote_id, data, grado=1):
        super().__init__(lote_id, data)
        self.grado = grado
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=self.grado)
        self.fecha_ordinal_min = None  # Almacenar el valor mínimo de la fecha ordinal

    def entrenar_modelo(self):
        promedios = self.calcular_promedios()
        promedios['Fecha_ordinal'] = promedios['Fecha'].map(pd.Timestamp.toordinal)
        self.fecha_ordinal_min = promedios['Fecha_ordinal'].min()  # Guardar el valor mínimo
        promedios['Fecha_ordinal_recentrada'] = promedios['Fecha_ordinal'] - self.fecha_ordinal_min # Recentrar
        X = promedios[['Fecha_ordinal_recentrada']]
        y = promedios['Peso']
        self.poly.fit(X)
        X_poly = self.poly.transform(X)
        self.model.fit(X_poly, y)

        # Visualizar los datos y la tendencia del modelo
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Datos')
        plt.plot(X, self.model.predict(X_poly), color='red', label='Modelo')
        plt.xlabel('Fecha Ordinal Recentrada') #Etiqueta correcta
        plt.ylabel('Peso (kg)')
        plt.title(f'Tendencia del modelo - Lote {self.lote_id}')
        plt.xticks(ticks=X['Fecha_ordinal_recentrada'], labels=promedios['Fecha'].dt.strftime('%Y-%m-%d'), rotation=45) #Tick labels correctos
        plt.legend()
        plt.grid(True)
        plt.show()

    def predecir_pesos(self, fecha_inicio, fecha_fin):
        fecha_inicio = pd.to_datetime(fecha_inicio)
        fecha_fin = pd.to_datetime(fecha_fin)
        fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
        fechas_ordinal = pd.DataFrame(fechas.map(pd.Timestamp.toordinal), columns=['Fecha_ordinal'])
        fechas_ordinal['Fecha_ordinal_recentrada'] = fechas_ordinal['Fecha_ordinal'] - self.fecha_ordinal_min #Recentrar
        X = fechas_ordinal[['Fecha_ordinal_recentrada']]
        fechas_ordinal_poly = self.poly.transform(X)
        pesos_predichos = self.model.predict(fechas_ordinal_poly)
        return pd.DataFrame({'Fecha': fechas, 'Peso_Predicho': pesos_predichos})

    def graficar_predicciones(self, fecha_inicio, fecha_fin):
        predicciones = self.predecir_pesos(fecha_inicio, fecha_fin)
        plt.figure(figsize=(10, 6))
        plt.plot(predicciones['Fecha'], predicciones['Peso_Predicho'], marker='o', label='Predicción')
        plt.xlabel('Fecha')
        plt.ylabel('Peso Predicho (kg)')
        plt.title(f'Predicción del peso por fecha - Lote {self.lote_id}')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

