import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuración
CSV_FILE = Path(__file__).parent / 'datos_sinteticos.csv'

# Configurar estilo de gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data(filepath):
    """Carga el archivo CSV"""
    try:
        df = pd.read_csv(filepath)
        print(f"[OK] Datos cargados exitosamente: {len(df)} registros")
        return df
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo {filepath}")
        return None

def print_section(title):
    """Imprime un encabezado de sección"""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}\n")

def analisis_general(df):
    """Análisis general del dataset"""
    print_section("1. INFORMACIÓN GENERAL DEL DATASET")
    print(f"Total de registros: {len(df)}")
    print(f"Total de columnas: {len(df.columns)}")
    print(f"\nColumnas del dataset:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

def analisis_tipos_datos(df):
    """Tabla de tipos de datos"""
    print_section("2. TIPOS DE DATOS")
    print(df.dtypes)

def analisis_estadistico(df):
    """Resumen estadístico de variables numéricas"""
    print_section("3. RESUMEN ESTADÍSTICO")
    print(df.describe().round(2))

def analisis_por_plataforma(df):
    """Análisis agregado por plataforma"""
    print_section("4. ANÁLISIS POR PLATAFORMA")
    
    plataformas = df.groupby('plataforma').agg({
        'campana_id': 'count',
        'presupuesto_diario': 'sum',
        'impresiones': 'sum',
        'clicks': 'sum',
        'conversiones': 'sum',
        'costo_total': 'sum',
        'revenue_generado': 'sum',
        'roas': 'mean'
    }).round(2)
    
    plataformas.columns = ['Campañas', 'Presupuesto Total', 'Impresiones', 
                           'Clicks', 'Conversiones', 'Costo Total', 'Revenue', 'ROAS Promedio']
    print(plataformas.to_string())

def analisis_por_tipo_campana(df):
    """Análisis agregado por tipo de campaña"""
    print_section("5. ANÁLISIS POR TIPO DE CAMPAÑA")
    
    tipos = df.groupby('tipo_campana').agg({
        'campana_id': 'count',
        'clicks': 'sum',
        'conversiones': 'sum',
        'conversion_rate': 'mean',
        'roas': 'mean'
    }).round(2)
    
    tipos.columns = ['Campañas', 'Total Clicks', 'Conversiones', 
                     'Conversion Rate Promedio', 'ROAS Promedio']
    print(tipos.to_string())

def analisis_por_edad(df):
    """Análisis agregado por grupo de edad"""
    print_section("6. ANÁLISIS POR GRUPO DE EDAD")
    
    edades = df.groupby('audiencia_objetivo').agg({
        'campana_id': 'count',
        'conversiones': 'sum',
        'costo_total': 'sum',
        'revenue_generado': 'sum',
        'roas': 'mean'
    }).round(2)
    
    edades.columns = ['Campañas', 'Conversiones', 'Costo', 'Revenue', 'ROAS Promedio']
    print(edades.to_string())

def indicadores_clave(df):
    """Cálculo e impresión de KPIs principales"""
    print_section("7. INDICADORES CLAVE (KPIs)")
    
    ctr_promedio = df['ctr'].mean()
    conversion_rate_promedio = df['conversion_rate'].mean()
    cpc_promedio = df['cpc'].mean()
    cpa_promedio = df['cpa'].mean()
    roas_promedio = df['roas'].mean()
    presupuesto_total = df['presupuesto_diario'].sum()
    costo_total = df['costo_total'].sum()
    revenue_total = df['revenue_generado'].sum()
    ganancia_neta = revenue_total - costo_total
    
    print(f"CTR (Click-Through Rate) Promedio:        {ctr_promedio:.2f}%")
    print(f"Conversion Rate Promedio:                 {conversion_rate_promedio:.2f}%")
    print(f"CPC (Costo por Click) Promedio:           ${cpc_promedio:.2f}")
    print(f"CPA (Costo por Adquisición) Promedio:     ${cpa_promedio:.2f}")
    print(f"ROAS (Return on Ad Spend) Promedio:       {roas_promedio:.2f}x")
    print(f"\nPresupuesto Total Invertido:              ${presupuesto_total:,.2f}")
    print(f"Costo Total Real:                         ${costo_total:,.2f}")
    print(f"Revenue Generado:                         ${revenue_total:,.2f}")
    print(f"Ganancia Neta:                            ${ganancia_neta:,.2f}")
    
    return {
        'ctr': ctr_promedio,
        'conversion_rate': conversion_rate_promedio,
        'cpc': cpc_promedio,
        'cpa': cpa_promedio,
        'roas': roas_promedio,
        'presupuesto_total': presupuesto_total,
        'costo_total': costo_total,
        'revenue_total': revenue_total,
        'ganancia_neta': ganancia_neta
    }

def campanas_top(df, n=3):
    """Muestra las TOP N campañas con mejor ROAS"""
    print_section(f"8. TOP {n} CAMPAÑAS POR ROAS")
    
    top = df.nlargest(n, 'roas')[['campana_id', 'plataforma', 'tipo_campana', 
                                    'roas', 'revenue_generado', 'costo_total']]
    print(top.to_string(index=False))

def campanas_bottom(df, n=3):
    """Muestra las BOTTOM N campañas con peor ROAS"""
    print_section(f"9. BOTTOM {n} CAMPAÑAS POR ROAS")
    
    bottom = df.nsmallest(n, 'roas')[['campana_id', 'plataforma', 'tipo_campana', 
                                        'roas', 'revenue_generado', 'costo_total']]
    print(bottom.to_string(index=False))

def validacion_datos(df):
    """Valida la integridad del dataset"""
    print_section("10. VALIDACIÓN DE DATOS")
    
    print("Valores faltantes por columna:")
    faltantes = df.isnull().sum()
    print(faltantes)
    
    total_faltantes = faltantes.sum()
    if total_faltantes == 0:
        print("\n[OK] No hay valores faltantes - Datos íntegros")
    else:
        print(f"\n[ERROR] Hay {total_faltantes} valores faltantes a revisar")
    
    return total_faltantes

def recomendaciones(df, kpis):
    """Genera recomendaciones basadas en el análisis"""
    print_section("11. RECOMENDACIONES ESTRATÉGICAS")
    
    # Mejor plataforma por ROAS
    mejor_plataforma = df.groupby('plataforma')['roas'].mean().idxmax()
    mejor_roas = df.groupby('plataforma')['roas'].mean().max()
    
    # Peor plataforma por ROAS
    peor_plataforma = df.groupby('plataforma')['roas'].mean().idxmin()
    peor_roas = df.groupby('plataforma')['roas'].mean().min()
    
    # Mejor audiencia por ROAS
    mejor_edad = df.groupby('audiencia_objetivo')['roas'].mean().idxmax()
    mejor_edad_roas = df.groupby('audiencia_objetivo')['roas'].mean().max()
    
    # Peor audiencia por ROAS
    peor_edad = df.groupby('audiencia_objetivo')['roas'].mean().idxmin()
    peor_edad_roas = df.groupby('audiencia_objetivo')['roas'].mean().min()
    
    print(f"1. PLATAFORMAS:")
    print(f"   [MEJOR] Mejor rendimiento: {mejor_plataforma} (ROAS {mejor_roas:.2f}x)")
    print(f"   [PEOR] Peor rendimiento: {peor_plataforma} (ROAS {peor_roas:.2f}x)")
    print(f"   [ACCION] Aumentar inversión en {mejor_plataforma} y revisar estrategia en {peor_plataforma}")
    
    print(f"\n2. PÚBLICO OBJETIVO:")
    print(f"   [MEJOR] Mejor rendimiento: {mejor_edad} años (ROAS {mejor_edad_roas:.2f}x)")
    print(f"   [PEOR] Peor rendimiento: {peor_edad} años (ROAS {peor_edad_roas:.2f}x)")
    print(f"   [ACCION] Maximizar campañas para {mejor_edad} años, optimizar o pausar {peor_edad} años")
    
    print(f"\n3. MÉTRICAS CLAVE:")
    if kpis['roas'] > 5:
        print(f"   [OK] ROAS promedio {kpis['roas']:.2f}x es saludable (>5x es excelente)")
    else:
        print(f"   [ALERTA] ROAS promedio {kpis['roas']:.2f}x está por debajo de lo ideal (objetivo >5x)")
    
    if kpis['conversion_rate'] > 3:
        print(f"   [OK] Conversion Rate {kpis['conversion_rate']:.2f}% es respectable")
    else:
        print(f"   [ALERTA] Conversion Rate {kpis['conversion_rate']:.2f}% necesita mejora")
    
    print(f"\n4. RENTABILIDAD:")
    margen = (kpis['ganancia_neta'] / kpis['revenue_total']) * 100
    print(f"   [OK] Ganancia: ${kpis['ganancia_neta']:,.2f}")
    print(f"   [OK] Margen de ganancia: {margen:.1f}%")

def generar_graficas(df, kpis):
    """Genera un conjunto completo de gráficas analíticas"""
    print_section("GENERANDO GRÁFICAS")
    
    # Crear figura con 6 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. ROAS por Plataforma
    ax1 = plt.subplot(2, 3, 1)
    roas_plataforma = df.groupby('plataforma')['roas'].mean().sort_values(ascending=False)
    colors = ['green' if x > 5 else 'orange' for x in roas_plataforma.values]
    roas_plataforma.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
    ax1.set_title('ROAS Promedio por Plataforma', fontsize=12, fontweight='bold')
    ax1.set_ylabel('ROAS (x)', fontsize=10)
    ax1.set_xlabel('Plataforma', fontsize=10)
    ax1.axhline(y=5, color='red', linestyle='--', label='Meta (5x)', alpha=0.7)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Revenue por Plataforma
    ax2 = plt.subplot(2, 3, 2)
    revenue_plataforma = df.groupby('plataforma')['revenue_generado'].sum().sort_values(ascending=False)
    revenue_plataforma.plot(kind='bar', ax=ax2, color='steelblue', edgecolor='black')
    ax2.set_title('Revenue Generado por Plataforma', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Revenue ($)', fontsize=10)
    ax2.set_xlabel('Plataforma', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Conversiones por Tipo de Campaña
    ax3 = plt.subplot(2, 3, 3)
    conversiones_tipo = df.groupby('tipo_campana')['conversiones'].sum().sort_values(ascending=False)
    ax3.pie(conversiones_tipo, labels=conversiones_tipo.index, autopct='%1.1f%%', 
            startangle=90, colors=sns.color_palette('Set2', len(conversiones_tipo)))
    ax3.set_title('Distribución de Conversiones\npor Tipo de Campaña', fontsize=12, fontweight='bold')
    
    # 4. ROAS por Grupo de Edad
    ax4 = plt.subplot(2, 3, 4)
    roas_edad = df.groupby('audiencia_objetivo')['roas'].mean().sort_values(ascending=False)
    colors_edad = ['darkgreen' if x > 10 else 'green' if x > 5 else 'orange' for x in roas_edad.values]
    roas_edad.plot(kind='barh', ax=ax4, color=colors_edad, edgecolor='black')
    ax4.set_title('ROAS Promedio por Grupo de Edad', fontsize=12, fontweight='bold')
    ax4.set_xlabel('ROAS (x)', fontsize=10)
    ax4.axvline(x=5, color='red', linestyle='--', alpha=0.7)
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Revenue vs Costo por Campaña
    ax5 = plt.subplot(2, 3, 5)
    top_campanas = df.nlargest(6, 'revenue_generado')
    x_pos = np.arange(len(top_campanas))
    width = 0.35
    bars1 = ax5.bar(x_pos - width/2, top_campanas['revenue_generado'], width, label='Revenue', color='lightgreen', edgecolor='black')
    bars2 = ax5.bar(x_pos + width/2, top_campanas['costo_total'], width, label='Costo', color='lightcoral', edgecolor='black')
    ax5.set_title('Top 6 Campañas: Revenue vs Costo', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Monto ($)', fontsize=10)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(top_campanas['campana_id'], rotation=45, ha='right')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Impresiones vs Clicks por Plataforma
    ax6 = plt.subplot(2, 3, 6)
    impresiones_plataforma = df.groupby('plataforma')['impresiones'].sum()
    clicks_plataforma = df.groupby('plataforma')['clicks'].sum()
    x_pos = np.arange(len(impresiones_plataforma))
    width = 0.35
    ax6.bar(x_pos - width/2, impresiones_plataforma, width, label='Impresiones', color='lightblue', edgecolor='black')
    ax6.bar(x_pos + width/2, clicks_plataforma, width, label='Clicks', color='orange', edgecolor='black')
    ax6.set_title('Impresiones vs Clicks por Plataforma', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cantidad', fontsize=10)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(impresiones_plataforma.index, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'analisis_graficas.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráficas guardadas en: analisis_graficas.png")
    plt.show()

def generar_metricas_kpi(df, kpis):
    """Genera gráficas de métricas clave"""
    print_section("GRÁFICAS DE MÉTRICAS CLAVE")
    
    fig = plt.figure(figsize=(14, 8))
    
    # 1. Gauge de ROAS
    ax1 = plt.subplot(2, 3, 1)
    ax1.barh(['ROAS'], [kpis['roas']], color='green' if kpis['roas'] > 5 else 'orange', edgecolor='black')
    ax1.set_xlim(0, 10)
    ax1.set_title('ROAS Promedio', fontsize=12, fontweight='bold')
    ax1.text(kpis['roas'] + 0.2, 0, f"{kpis['roas']:.2f}x", va='center', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. CTR
    ax2 = plt.subplot(2, 3, 2)
    ax2.barh(['CTR'], [kpis['ctr']], color='steelblue', edgecolor='black')
    ax2.set_xlim(0, 15)
    ax2.set_title('CTR Promedio', fontsize=12, fontweight='bold')
    ax2.set_xlabel('%')
    ax2.text(kpis['ctr'] + 0.3, 0, f"{kpis['ctr']:.2f}%", va='center', fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Conversion Rate
    ax3 = plt.subplot(2, 3, 3)
    ax3.barh(['Conv. Rate'], [kpis['conversion_rate']], color='purple', edgecolor='black')
    ax3.set_xlim(0, 10)
    ax3.set_title('Conversion Rate Promedio', fontsize=12, fontweight='bold')
    ax3.set_xlabel('%')
    ax3.text(kpis['conversion_rate'] + 0.2, 0, f"{kpis['conversion_rate']:.2f}%", va='center', fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. CPC
    ax4 = plt.subplot(2, 3, 4)
    ax4.barh(['CPC'], [kpis['cpc']], color='coral', edgecolor='black')
    ax4.set_xlim(0, 2)
    ax4.set_title('Costo por Click (CPC)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('$')
    ax4.text(kpis['cpc'] + 0.05, 0, f"${kpis['cpc']:.2f}", va='center', fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. CPA
    ax5 = plt.subplot(2, 3, 5)
    ax5.barh(['CPA'], [kpis['cpa']], color='chocolate', edgecolor='black')
    ax5.set_xlim(0, 50)
    ax5.set_title('Costo por Adquisición (CPA)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('$')
    ax5.text(kpis['cpa'] + 1, 0, f"${kpis['cpa']:.2f}", va='center', fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Margin de Ganancia
    ax6 = plt.subplot(2, 3, 6)
    margen = (kpis['ganancia_neta'] / kpis['revenue_total']) * 100
    ax6.barh(['Margen'], [margen], color='darkgreen', edgecolor='black')
    ax6.set_xlim(0, 100)
    ax6.set_title('Margen de Ganancia', fontsize=12, fontweight='bold')
    ax6.set_xlabel('%')
    ax6.text(margen + 2, 0, f"{margen:.1f}%", va='center', fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'metricas_kpi.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráficas de KPIs guardadas en: metricas_kpi.png")
    plt.show()

def generar_comparativa_campanas(df):
    """Genera gráficas de desempeño de campañas"""
    print_section("COMPARATIVA DE CAMPAÑAS")
    
    fig = plt.figure(figsize=(16, 8))
    
    # Ordenar por ROAS
    df_sorted = df.sort_values('roas', ascending=True)
    
    # 1. ROAS por campaña
    ax1 = plt.subplot(2, 2, 1)
    colors = ['red' if x < 2 else 'orange' if x < 5 else 'lightgreen' if x < 10 else 'darkgreen' for x in df_sorted['roas']]
    ax1.barh(df_sorted['campana_id'], df_sorted['roas'], color=colors, edgecolor='black')
    ax1.set_title('ROAS por Campaña (Ordenado)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('ROAS (x)')
    ax1.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='Meta (5x)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Scatter: Costo vs Revenue
    ax2 = plt.subplot(2, 2, 2)
    plataformas = df['plataforma'].unique()
    colors_map = {'Facebook Ads': 'blue', 'Instagram Ads': 'pink', 'LinkedIn Ads': 'navy', 'TikTok Ads': 'black'}
    for plat in plataformas:
        df_plat = df[df['plataforma'] == plat]
        ax2.scatter(df_plat['costo_total'], df_plat['revenue_generado'], 
                   label=plat, s=100, alpha=0.6, edgecolor='black')
    ax2.plot(df['costo_total'], df['costo_total'], 'r--', alpha=0.5, label='Break-even')
    ax2.set_title('Costo vs Revenue (por Campaña)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Costo Total ($)')
    ax2.set_ylabel('Revenue Generado ($)')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Conversiones por campaña
    ax3 = plt.subplot(2, 2, 3)
    df_conv = df.sort_values('conversiones', ascending=True)
    colors_conv = [df_conv.loc[idx, 'plataforma'] for idx in df_conv.index]
    color_map_conv = {'Facebook Ads': 'blue', 'Instagram Ads': 'pink', 'LinkedIn Ads': 'navy', 'TikTok Ads': 'black'}
    colors_list = [color_map_conv.get(c, 'gray') for c in colors_conv]
    ax3.barh(df_conv['campana_id'], df_conv['conversiones'], color=colors_list, edgecolor='black')
    ax3.set_title('Conversiones por Campaña', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Conversiones')
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. CTR por campaña
    ax4 = plt.subplot(2, 2, 4)
    df_ctr = df.sort_values('ctr', ascending=True)
    colors_ctr = [color_map_conv.get(df_ctr.loc[idx, 'plataforma'], 'gray') for idx in df_ctr.index]
    ax4.barh(df_ctr['campana_id'], df_ctr['ctr'], color=colors_ctr, edgecolor='black')
    ax4.set_title('CTR por Campaña', fontsize=12, fontweight='bold')
    ax4.set_xlabel('CTR (%)')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'comparativa_campanas.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráficas de campañas guardadas en: comparativa_campanas.png")
    plt.show()

def main():
    """Función principal - ejecuta todo el análisis"""
    df = load_data(CSV_FILE)
    
    if df is None:
        return
    
    # Ejecutar todos los análisis
    analisis_general(df)
    analisis_tipos_datos(df)
    analisis_estadistico(df)
    analisis_por_plataforma(df)
    analisis_por_tipo_campana(df)
    analisis_por_edad(df)
    kpis = indicadores_clave(df)
    campanas_top(df, 3)
    campanas_bottom(df, 3)
    validacion_datos(df)
    recomendaciones(df, kpis)
    
    # Generar gráficas
    generar_graficas(df, kpis)
    generar_metricas_kpi(df, kpis)
    generar_comparativa_campanas(df)
    
    print("\n" + "=" * 80)
    print("FIN DEL ANÁLISIS")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
