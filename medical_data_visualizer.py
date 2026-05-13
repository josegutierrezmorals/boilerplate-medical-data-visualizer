import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar datos
df = pd.read_csv('medical_examination.csv')

# 2. Columna overweight — BMI = peso(kg) / altura(m)²
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalizar cholesterol y gluc: 1→0 (normal), >1→1 (malo)
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc']        = (df['gluc'] > 1).astype(int)


def draw_cat_plot():
    # 4. Crear DataFrame en formato largo con pd.melt
    df_cat = pd.melt(
        df,
        id_vars='cardio',
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5. Agrupar y contar, renombrar columna para catplot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    df_cat.rename(columns={0: 'total'}, inplace=True)

    # 6. Dibujar catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 7. Guardar y retornar
    fig.savefig('catplot.png')
    return fig


def draw_heat_map():
    # 8. Limpiar datos incorrectos
    df_heat = df[
        (df['ap_lo']  <= df['ap_hi'])                          &
        (df['height'] >= df['height'].quantile(0.025))         &
        (df['height'] <= df['height'].quantile(0.975))         &
        (df['weight'] >= df['weight'].quantile(0.025))         &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 9. Matriz de correlación
    corr = df_heat.corr()

    # 10. Máscara para triángulo superior
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # 11. Figura matplotlib
    fig, ax = plt.subplots(figsize=(12, 9))

    # 12. Heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )

    # 13. Guardar y retornar
    fig.savefig('heatmap.png')
    return fig
