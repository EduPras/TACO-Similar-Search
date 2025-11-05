import pandas as pd
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import json

class FoodData:
    def __init__(self, csv_path: str, is_ptbr = False):
        self.df = pd.read_csv(csv_path)
        self.df_original = self.df.copy()
        self.client = chromadb.Client(Settings())
        self.collection_name = "FoodDB"
        self.collection = None
        
        # # If you are using the original taco_table.csv (pt-BR) you may remove these columns
        if is_ptbr:
            print('Loading for pt-br')
            drop_cols = [
                'Ashes (g)', 'Calcium (mg)',
                'Magnesium (mg)', 'Manganese (mg)', 'Phosphorus (mg)', 'Iron (mg)',
                'Sodium (mg)', 'Potassium (mg)', 'Copper (mg)', 'Zinc (mg)',
                'Retinol (µg)', 'RE (µg)', 'RAE (µg)', 'Thiamine (mg)',
                'Riboflavin (mg)', 'Pyridoxine (mg)', 'Niacin (mg)', 'Vitamin C (mg)',
                'Humidity (%)',  'Energy (kJ)',  'Cholesterol (mg)'
            ]
            self.df.drop(columns=drop_cols, inplace=True)
            self.df_original.drop(columns=drop_cols, inplace=True)

        self.df.fillna(0, inplace=True)

        if self.df['id'].dtype != 'string':
            self.df['id'] = self.df['id'].astype(str)
        
        scaler = MinMaxScaler()
        self.numeric_cols = self.df.select_dtypes(include=['number']).columns
        self.df[self.numeric_cols] = scaler.fit_transform(self.df[self.numeric_cols])
        
    
    def create_vector_db(self):
        if self.collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Creating vector database..."):
            doc_id = str(idx)
            metadata = {"Food Description": row["Food Description"]}
            embedding = row[self.numeric_cols].tolist()
            document = self.df_original.loc[idx].to_json()
            self.collection.add(documents=[document], ids=[doc_id], metadatas=[metadata], embeddings=[embedding])

    def query(self, food_id: str, n: int =5) -> pd.DataFrame:
        idx_list = self.df.index[self.df['id'] == str(food_id)].tolist()
        print(idx_list)
        print(self.df['id'])
        if not idx_list:
            raise ValueError(f"Food ID {food_id} not found.")
        idx = idx_list[0]
        query_embedding = self.df.loc[idx, self.numeric_cols].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["embeddings", "documents", "metadatas"]
        )
        return pd.DataFrame(json.loads(x) for x in results['documents'][0])
        
        
# Load DataFrame
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'
    
languages = {
    'en': 'taco_table_en.csv', 
    'pt-br': 'taco_table.csv'
}
selected_language = st.sidebar.selectbox(
    'Select a language:',
    list(languages.keys())
)
DF_PATH = languages[selected_language]
if selected_language != st.session_state['language']:
    st.session_state['language'] = selected_language
    DF_PATH = languages[selected_language]
    is_ptbr = False if selected_language == 'en' else True
    st.session_state['data_handler'] = FoodData(DF_PATH, is_ptbr)
    st.session_state['data_handler'].create_vector_db()

if 'data_handler' not in st.session_state:
    is_ptbr = False if selected_language == 'en' else True
    st.session_state['data_handler'] = FoodData(DF_PATH, is_ptbr)
    st.session_state['data_handler'].create_vector_db()
data_handler = st.session_state['data_handler']

st.title("Food Similarity Search")
st.markdown("Search for a food and find similar foods using the vector database.")
st.set_page_config(layout="wide")

food_options = data_handler.df[['id', 'Food Description']].values.tolist()
food_labels = [f"{desc} (ID: {fid})" for fid, desc in food_options]
selected_idx = st.multiselect(
    "Select a food:", options=range(len(food_labels)), format_func=lambda i: food_labels[i], max_selections=1)
st.space("large")
print(selected_idx)
if selected_idx is not None and len(selected_idx) > 0 :
    item = selected_idx[0]
    selected_food_id, selected_food_desc = food_options[item]
    st.write(f"Selected Food: {selected_food_desc} (ID: {selected_food_id})")
    st.dataframe(data_handler.query(str(selected_food_id)), width='stretch')