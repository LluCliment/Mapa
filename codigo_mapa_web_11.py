import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import glob
import os
import unidecode # Para quitar acentos y normalizar
import re # Para expresiones regulares

# --- Configuración de la Página de Streamlit ---
# ... (sin cambios) ...
st.set_page_config(layout="wide", page_title="Visualizador de Mapas CCAA")
st.title("🗺️ Visualizador de Mapas por Comunidad Autónoma")
st.markdown("""
    Selecciona un conjunto de datos y el grupo de sexo para ver su distribución 
    geográfica en un mapa de España por Comunidades Autónomas.
""")

# --- Función de Normalización de Nombres de CCAA ---
# ... (sin cambios) ...
def normalize_cca_name(name):
    if pd.isna(name):
        return None
    s = unidecode.unidecode(str(name).lower().strip())
    if s == "total": return None
    specific_mappings = {
        "principado de asturias": "asturias", "illes balears": "balears", "islas baleares": "balears",
        "canarias": "canarias", "islas canarias": "canarias", "castilla - la mancha": "castilla-la mancha",
        "castilla y leon": "castilla y leon", "catalunya": "cataluna", "comunitat valenciana": "valencia",
        "comunidad valenciana": "valencia", "madrid comunidad de": "madrid", "comunidad de madrid": "madrid",
        "murcia region de": "murcia", "region de murcia": "murcia", "navarra comunidad foral de": "navarra",
        "comunidad foral de navarra": "navarra", "pais vasco": "pais vasco", "euskadi / pais vasco": "pais vasco",
        "rioja la": "rioja", "la rioja": "rioja", "ceuta": "ceuta", "ciudad autonoma de ceuta": "ceuta",
        "melilla": "melilla", "ciudad autonoma de melilla": "melilla", 
        "andalucia": "andalucia", "aragon": "aragon", "asturias (principado de)": "asturias", 
        "balears (illes)": "balears", "cantabria": "cantabria", "extremadura": "extremadura", 
        "galicia": "galicia", "madrid (comunidad de)": "madrid", "murcia (region de)": "murcia",
        "navarra (comunidad foral de)": "navarra", "rioja (la)": "rioja"
    }
    if s in specific_mappings: s = specific_mappings[s]
    s = re.sub(r'\s*\([^)]*\)', '', s)
    s = s.replace('-', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    final_canonical_map = {
        "castilla la mancha": "castilla-la mancha", "castilla leon": "castilla y leon", "valencia": "valencia"
    }
    if s in final_canonical_map: s = final_canonical_map[s]
    if s == "baleares": s = "balears"
    return s.strip() if s else None

# --- Carga y Caché del GeoJSON ---
# ... (sin cambios) ...
@st.cache_data
def load_geojson(geojson_path="spain-communities.geojson"):
    try:
        gdf = gpd.read_file(geojson_path)
        name_column_geojson = 'name'
        if name_column_geojson not in gdf.columns:
            st.error(f"Error: Columna '{name_column_geojson}' no encontrada en GeoJSON '{geojson_path}'.")
            return None 
        gdf['normalized_cca_name'] = gdf[name_column_geojson].apply(normalize_cca_name)
        gdf.dropna(subset=['normalized_cca_name'], inplace=True)
        return gdf
    except Exception as e:
        st.error(f"Error al cargar GeoJSON desde '{geojson_path}': {e}")
        return None

# --- Función para Cargar y Procesar un CSV específico ---
# ... (sin cambios) ...
@st.cache_data
def load_and_process_csv(csv_path, selected_sex_normalized):
    try:
        try:
            df = pd.read_csv(csv_path, decimal=',', sep=';')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, decimal=',', sep=';', encoding='latin1')
        
        df.columns = [unidecode.unidecode(col.lower().strip().replace(" ", "_")) for col in df.columns]
        expected_cols_internal = ['sexo', 'comunidad_autonoma', 'total']
        if not all(col in df.columns for col in expected_cols_internal): return None
        
        df = df[~df['comunidad_autonoma'].astype(str).str.contains("Total", case=False, na=False)]
        if df.empty: return None
        
        if df['total'].dtype == 'object':
            df['total'] = df['total'].astype(str).str.replace('%', '', regex=False)
            df['total'] = df['total'].str.replace(',', '.', regex=False)
        df['total'] = pd.to_numeric(df['total'], errors='coerce')
        if df['total'].isnull().all(): return None
            
        if 'sexo' in df.columns:
            df['sexo'] = df['sexo'].astype(str).apply(lambda x: unidecode.unidecode(x.lower().strip()))
        
        df_sexo_filtrado = df[df['sexo'] == selected_sex_normalized].copy()
        if df_sexo_filtrado.empty: return None
            
        df_sexo_filtrado['normalized_cca_name_csv'] = df_sexo_filtrado['comunidad_autonoma'].apply(normalize_cca_name)
        df_sexo_filtrado.dropna(subset=['normalized_cca_name_csv'], inplace=True)
        if df_sexo_filtrado.empty: return None
            
        rows_to_add_cm = []
        indices_to_drop_cm = []
        for index in df_sexo_filtrado.index:
            row = df_sexo_filtrado.loc[index]
            if row['normalized_cca_name_csv'] == 'ceuta y melilla':
                ceuta_row = row.copy(); ceuta_row['comunidad_autonoma'] = 'Ceuta'; ceuta_row['normalized_cca_name_csv'] = normalize_cca_name('Ceuta'); rows_to_add_cm.append(ceuta_row)
                melilla_row = row.copy(); melilla_row['comunidad_autonoma'] = 'Melilla'; melilla_row['normalized_cca_name_csv'] = normalize_cca_name('Melilla'); rows_to_add_cm.append(melilla_row)
                indices_to_drop_cm.append(index)
        if indices_to_drop_cm:
            df_sexo_filtrado = df_sexo_filtrado.drop(indices_to_drop_cm)
            if rows_to_add_cm: 
                df_sexo_filtrado = pd.concat([df_sexo_filtrado, pd.DataFrame(rows_to_add_cm)], ignore_index=True)
        df_sexo_filtrado = df_sexo_filtrado.drop_duplicates(subset=['normalized_cca_name_csv'], keep='first')
        
        return df_sexo_filtrado
    except Exception as e:
        print(f"Error procesando CSV '{os.path.basename(csv_path)}' para sexo '{selected_sex_normalized}': {e}")
        return None

# --- Lógica Principal de la App Streamlit ---
gdf_spain = load_geojson() 
if gdf_spain is None:
    st.error("No se pudo cargar el archivo GeoJSON base. La aplicación no puede continuar.")
    st.stop()

csv_folder = "."
csv_files_paths = glob.glob(os.path.join(csv_folder, "*.csv"))
if not csv_files_paths:
    st.warning("No se encontraron archivos CSV en la carpeta actual.")
    st.stop()

csv_files_dict = {
    os.path.splitext(os.path.basename(path))[0].replace("_", " ").capitalize(): path 
    for path in csv_files_paths
}

col1, col2 = st.columns(2)
with col1:
    selected_csv_name = st.selectbox(
        "1. Selecciona un conjunto de datos:", 
        options=list(csv_files_dict.keys()),
        index=0
    )
sex_options_display = ["Ambos sexos", "Varones", "Mujeres"]
sex_options_normalized_map = {
    "Ambos sexos": "ambos sexos", "Varones": "varones", "Mujeres": "mujeres"
}
# Mapeo de sexos a cmaps
sex_to_cmap_map = {
    "ambos sexos": "YlOrRd", # Amarillo-Naranja-Rojo
    "varones": "Blues",      # Escala de Azules
    "mujeres": "RdPu"        # Escala de Rojos-Púrpuras (o "Reds", "Purples", "BuPu")
}

with col2:
    selected_sex_display = st.selectbox(
        "2. Selecciona el grupo de sexo:",
        options=sex_options_display,
        index=0
    )
selected_sex_normalized = sex_options_normalized_map[selected_sex_display]
selected_cmap = sex_to_cmap_map[selected_sex_normalized] # Obtener el cmap según el sexo

if selected_csv_name and selected_sex_display:
    selected_csv_path = csv_files_dict[selected_csv_name]
    st.subheader(f"Visualización: {selected_csv_name} ({selected_sex_display})")

    df_processed_csv = load_and_process_csv(selected_csv_path, selected_sex_normalized)

    if df_processed_csv is not None and not df_processed_csv.empty:
        merged_gdf = gdf_spain.merge(
            df_processed_csv,
            left_on='normalized_cca_name',
            right_on='normalized_cca_name_csv',
            how='left'
        )

        fig, ax_main = plt.subplots(1, 1, figsize=(9, 7.5))
        canarias_gdf = merged_gdf[merged_gdf['normalized_cca_name'] == 'canarias']
        
        missing_kwds = {"color": "lightgrey", "edgecolor": "black", "hatch": "///", "label": "Sin datos"}
        
        valid_totals = merged_gdf['total'].dropna()
        vmin, vmax = None, None
        if not valid_totals.empty:
            vmin = valid_totals.min()
            vmax = valid_totals.max()
        norm = plt.Normalize(vmin=vmin, vmax=vmax) if vmin is not None and vmax is not None else None

        # Usar el selected_cmap dinámicamente
        merged_gdf.plot(
            column='total', cmap=selected_cmap, linewidth=0.8, ax=ax_main, edgecolor='0.7',
            legend=False, missing_kwds=missing_kwds, norm=norm
        )
        
        ax_main.set_xlim(-10.5, 4.5); ax_main.set_ylim(35.0, 44.3); ax_main.set_axis_off()

        inset_left = 0.05; inset_bottom = 0.15; inset_width = 0.18; inset_height = 0.22
        ax_canarias = fig.add_axes([inset_left, inset_bottom, inset_width, inset_height])
        
        if not canarias_gdf.empty:
            canarias_gdf.plot(
                column='total', cmap=selected_cmap, linewidth=0.7, ax=ax_canarias, edgecolor='0.4', 
                missing_kwds=missing_kwds, norm=norm
            )
        else:
            ax_canarias.text(0.5, 0.5, 'Canarias\n(s/d)', ha='center', va='center', transform=ax_canarias.transAxes, fontsize=6)
        
        ax_canarias.set_xticks([]); ax_canarias.set_yticks([]); ax_canarias.set_axis_off()
        for spine in ax_canarias.spines.values(): spine.set_edgecolor("black"); spine.set_linewidth(0.6)

        if norm:
            sm = plt.cm.ScalarMappable(cmap=selected_cmap, norm=norm); sm.set_array([]) 
            cbar_left = 0.25; cbar_width = 0.50
            cbar_axes_rect_centered = [cbar_left, 0.12, cbar_width, 0.03]
            cax = fig.add_axes(cbar_axes_rect_centered)
            cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', label="Porcentaje (%)")
            cbar.ax.tick_params(labelsize=7); cbar.set_label("Porcentaje (%)", size=8)

        st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.markdown("<p style='text-align: center; font-size: 0.85em; color: grey;'>Datos obtenidos de: Encuesta Nacional de Salud y Hábitos Sexuales (2003), Instituto Nacional de Estadística (INE).</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.9em; margin-top: 20px;'><strong>Autores:</strong><br>Lluc Climent Navarro<br>Raúl Company Martínez<br>Aleix Torró Abad<br>Hugo Cuartero Álvarez</p>", unsafe_allow_html=True)
        
        unmatched_csv_in_plot = df_processed_csv[~df_processed_csv['normalized_cca_name_csv'].isin(gdf_spain['normalized_cca_name'].unique())]
        unmatched_csv_in_plot = unmatched_csv_in_plot[unmatched_csv_in_plot['comunidad_autonoma'].notna() & (unmatched_csv_in_plot['comunidad_autonoma'].astype(str).str.strip() != '')]
        if not unmatched_csv_in_plot.empty:
            with st.expander("Aviso: Algunas CCAA del CSV no coincidieron con el mapa", expanded=False):
                for _, row_unmatched in unmatched_csv_in_plot.iterrows():
                    st.write(f"  - CSV: '{row_unmatched['comunidad_autonoma']}' (normalizado a: '{row_unmatched['normalized_cca_name_csv']}')")
    
    elif df_processed_csv is None:
         st.error(f"No se pudieron procesar los datos del archivo '{selected_csv_name}' para '{selected_sex_display}'. Asegúrate de que el archivo contiene datos para este grupo.")
    else:
        st.info(f"No hay datos válidos para mostrar en el mapa para '{selected_csv_name}' ({selected_sex_display}) después del filtrado y procesamiento.")
else:
    st.info("Por favor, selecciona un archivo CSV y un grupo de sexo.")