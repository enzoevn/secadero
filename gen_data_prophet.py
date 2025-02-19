import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

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
    def __init__(self, lote_id, data):
        super().__init__(lote_id, data)
        self.model = None

    def entrenar_modelo(self):
        promedios = self.calcular_promedios()
        promedios = promedios.rename(columns={'Fecha': 'ds', 'Peso': 'y'})

        # Inicializar y entrenar el modelo Prophet
        self.model = Prophet(
            seasonality_prior_scale=0.01,  # Ajustar el suavizado de la estacionalidad
            changepoint_prior_scale=0.1,  # Ajustar el suavizado de la tendencia
            seasonality_mode='additive' #Usar modelo aditivo
        )

        # self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5, prior_scale=0.02) #Añadir regularización
        self.model.fit(promedios)

        # Visualizar los datos y la tendencia del modelo (Opcional)
        future = self.model.make_future_dataframe(periods=0)  # Para graficar sobre los datos de entrenamiento
        forecast = self.model.predict(future)

        fig1 = self.model.plot(forecast)
        plt.title(f'Tendencia del modelo - Lote {self.lote_id}')
        plt.show()

        fig2 = self.model.plot_components(forecast)
        plt.show()

    def predecir_pesos(self, fecha_inicio, fecha_fin):
        fecha_inicio = pd.to_datetime(fecha_inicio)
        fecha_fin = pd.to_datetime(fecha_fin)
        fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
        future = pd.DataFrame(fechas, columns=['ds'])

        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat']].rename(columns={'ds': 'Fecha', 'yhat': 'Peso_Predicho'})

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