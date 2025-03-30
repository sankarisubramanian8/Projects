
import pymysql
import pandas as pd
import csv
def get_db_connection():
    connection = pymysql.connect(
        host="127.0.0.1",
        user="root",
        password="",
        port=3306
    )
    return connection

def data_cleaning():
    global csv_file_path 
    csv_file_path =r'E:\Project_2\Project_2\Scripts\ICRISAT-District Level Data - ICRISAT-District Level Data.csv'
    try:
        df = pd.read_csv(csv_file_path, delimiter=',', quoting=csv.QUOTE_MINIMAL, on_bad_lines='skip')
    except pd.errors.ParserError as e:
        df = pd.read_csv(csv_file_path, header=None)

    
    selected_columns = [
        'Year',
        'State Name',
        'Dist Name',
        'WHEAT PRODUCTION (1000 tons)',
        'OILSEEDS PRODUCTION (1000 tons)',
        'SUNFLOWER PRODUCTION (1000 tons)',
        'SUGARCANE PRODUCTION (1000 tons)',
        'RICE PRODUCTION (1000 tons)',
        'PEARL MILLET PRODUCTION (1000 tons)',
        'FINGER MILLET PRODUCTION (1000 tons)',
        'KHARIF SORGHUM PRODUCTION (1000 tons)',
        'RABI SORGHUM PRODUCTION (1000 tons)',
        'SORGHUM PRODUCTION (1000 tons)',
        'GROUNDNUT PRODUCTION (1000 tons)',
        'SOYABEAN PRODUCTION (1000 tons)',
        'SOYABEAN YIELD (Kg per ha)',
        'MAIZE AREA (1000 ha)',
        'WHEAT AREA (1000 ha)',
        'RICE YIELD (Kg per ha)',
        'WHEAT YIELD (Kg per ha)'
    ]

    new_df = df[selected_columns].copy()

    new_column_names = {
        'State Name': 'State',
        'Dist Name': 'District',
        'WHEAT PRODUCTION (1000 tons)': 'Wheat_prod',
        'OILSEEDS PRODUCTION (1000 tons)': 'Oilseed_prod',
        'SUNFLOWER PRODUCTION (1000 tons)': 'Sunflower_prod',
        'SUGARCANE PRODUCTION (1000 tons)': 'Sugarcane_prod',
        'RICE PRODUCTION (1000 tons)': 'Rice_prod',
        'PEARL MILLET PRODUCTION (1000 tons)': 'Pearl_millet_prod',
        'FINGER MILLET PRODUCTION (1000 tons)': 'Finger_millet_prod',
        'KHARIF SORGHUM PRODUCTION (1000 tons)': 'Kharif_sorghum_prod',
        'RABI SORGHUM PRODUCTION (1000 tons)': 'Rabi_sorghum_prod',
        'SORGHUM PRODUCTION (1000 tons)': 'Sorghum_prod',
        'GROUNDNUT PRODUCTION (1000 tons)': 'Groundnut_prod',
        'SOYABEAN PRODUCTION (1000 tons)': 'Soyabean_prod',
        'SOYABEAN YIELD (Kg per ha)': 'Soyabean_yield',
        'MAIZE AREA (1000 ha)': 'Maize_area',
        'WHEAT AREA (1000 ha)': 'Wheat_area',
        'RICE YIELD (Kg per ha)': 'Rice_yield',
        'WHEAT YIELD (Kg per ha)': 'Wheat_yield'
    }
    
    new_df = new_df.rename(columns=new_column_names)
    new_df.to_csv(csv_file_path, index=False)

    

def create_db_table_insert():
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Create database & table
        cursor.execute("CREATE DATABASE IF NOT EXISTS project_2")
        cursor.execute("USE project_2")
        
        cursor.execute("""
                        CREATE TABLE IF NOT EXISTS agri_data (
                            year INT,
                            state VARCHAR(100),
                            district VARCHAR(100),
                            wheat_prod FLOAT,
                            oilseed_prod FLOAT,
                            sunflower_prod FLOAT,
                            sugarcane_prod FLOAT,
                            rice_prod FLOAT,
                            pearl_millet_prod FLOAT,
                            finger_millet_prod FLOAT,
                            kharif_sorghum_prod FLOAT,
                            rabi_sorghum_prod FLOAT,
                            sorghum_prod FLOAT,
                            groundnut_prod FLOAT,
                            soyabean_prod FLOAT,
                            soyabean_yield FLOAT,
                            maize_area FLOAT,
                            wheat_area FLOAT,
                            rice_yield FLOAT,
                            wheat_yield FLOAT,
                            PRIMARY KEY (year, state, district))
        """)
        connection.commit()

        # Read  data
        csv_file_path =r'E:\Project_2\Project_2\Scripts\ICRISAT-District Level Data - ICRISAT-District Level Data.csv'
        try:
            # global csv_file_path 
            
            df = pd.read_csv(csv_file_path, on_bad_lines='skip')
        except pd.errors.ParserError as e:
            df = pd.read_csv(csv_file_path, header=None)

      
        required_columns = [
            'Year', 'State', 'District', 'Wheat_prod', 'Oilseed_prod',
            'Sunflower_prod', 'Sugarcane_prod', 'Rice_prod',
            'Pearl_millet_prod', 'Finger_millet_prod', 'Kharif_sorghum_prod',
            'Rabi_sorghum_prod', 'Sorghum_prod', 'Groundnut_prod',
            'Soyabean_prod', 'Soyabean_yield', 'Maize_area',
            'Wheat_area', 'Rice_yield', 'Wheat_yield'
        ]

       
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        
        data_tuples = [tuple(x) for x in df[required_columns].values]
        
        cursor.executemany("""
            INSERT INTO agri_data VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 
                                       %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, data_tuples)
        
        connection.commit()
        print(f"Successfully inserted {len(df)} records!")

    except Exception as e:
        print(f"Error: {str(e)}")
        connection.rollback()
    finally:
        cursor.close()
        connection.close()
def main():
    data_cleaning()    
    create_db_table_insert()
    
if __name__ == "__main__":
    main()
    