import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ucitavanje ociscenih podataka
df = pd.read_csv('cars_processed.csv')
print(df.info())

#1 Ispis broja mjerenja u dataframeu
print("\nBroj mjerenja:", len(df))

#2 Ispis tipova stupaca
print(df.dtypes)

#3 Automobil s najvećom cijenom
najskuplji_auto = df.loc[df['selling_price'].idxmax()]
cijena_kn = round(najskuplji_auto['selling_price'] * 1000, 2)
print("\nNajskuplji auto:", najskuplji_auto['name'], "(", "{:.2f}".format(cijena_kn), "kn)")

# Automobil s najmanjom cijenom
najjeftiniji_auto = df.loc[df['selling_price'].idxmin()]
cijena_knJEF = round(najjeftiniji_auto['selling_price'] * 1000, 2)
print("\nNajskuplji auto:", najjeftiniji_auto['name'], "(", "{:.2f}".format(cijena_knJEF), "kn)")

#4 Broj automobila proizvedenih 2012. godine
broj_automobila_2012 = df['year'].value_counts()[2012]
print("\nBroj automobila proizvedenih 2012. godine:", broj_automobila_2012)

#5 Automobil s najviše prijeđenih kilometara
najvise_km_auto = df.loc[df['km_driven'].idxmax()]
print("\nAutomobil s najviše prijeđenih kilometara:", najvise_km_auto['name'], "(", najvise_km_auto['km_driven'], "km)")

# Automobil s najmanje prijeđenih kilometara
najmanje_km_auto = df.loc[df['km_driven'].idxmin()]
print("\nAutomobil s najmanje prijeđenih kilometara:", najmanje_km_auto['name'], "(", najmanje_km_auto['km_driven'], "km)")

#6 Najčešći broj sjedala
najcesci_broj_sjedala = df['seats'].value_counts().idxmax()
print("\nNajčešći broj sjedala:", najcesci_broj_sjedala)

#7 Prosječna prijeđena kilometraža za automobile s dizel motorom
prosjecna_km_dizel = df.groupby('fuel')['km_driven'].mean()['Diesel']
print("\nProsječna prijeđena kilometraža za automobile s dizel motorom:", prosjecna_km_dizel, "km")

# Prosječna prijeđena kilometraža za automobile s benzinskim motorom
prosjecna_km_benzin = df.groupby('fuel')['km_driven'].mean()['Petrol']
print("\nProsječna prijeđena kilometraža za automobile s benzinskim motorom:", prosjecna_km_benzin, "km")

