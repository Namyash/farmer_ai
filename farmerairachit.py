import streamlit as st
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re

# Load data for schemes and legal rights
schemes_data = [
    {
        "id": "pm-kisan",
        "title": "PM-KISAN",
        "description": "Pradhan Mantri Kisan Samman Nidhi is a Central Sector scheme with 100% funding from Government of India.",
        "eligibility": "All landholding farmers' families, which have cultivable landholding in their names.",
        "benefits": "₹6000 per year disbursed in three equal installments of ₹2000 each every four months.",
        "applicationProcess": "Register through the official PM-KISAN portal or visit your local agriculture office."
    },
    {
        "id": "soil-health-card",
        "title": "Soil Health Card Scheme",
        "description": "Scheme to issue soil health cards to farmers with crop-wise recommendations for nutrients and fertilizers.",
        "eligibility": "All farmers with agricultural land.",
        "benefits": "Improved soil health, increased productivity, and reduced cultivation cost.",
        "applicationProcess": "Contact local agriculture department or Krishi Vigyan Kendra."
    }
]

legal_rights_data = [
    {
        "id": "land-ownership",
        "title": "Land Ownership Rights",
        "description": "Farmers have the right to own agricultural land and receive proper documentation for their property.",
        "relevantLaws": "Land Revenue Code, Registration Act",
        "applicability": "All farmers who own agricultural land"
    },
    {
        "id": "water-rights",
        "title": "Water Rights",
        "description": "Farmers have rights to access water resources for irrigation purposes.",
        "relevantLaws": "Water (Prevention and Control of Pollution) Act, Irrigation Acts",
        "applicability": "All farmers requiring water for agricultural purposes"
    }
]

translations = {
    'en': {
        'no_results': "I couldn't find specific information about your query. Please try asking about government schemes like PM-KISAN, PMFBY, or about land rights, water rights, or minimum support prices.",
        'found_info': "Here is some information related to your query about '{}':"
    },
    'hi': {
        'no_results': "मुझे आपके प्रश्न के बारे में विशिष्ट जानकारी नहीं मिली। कृपया PM-KISAN, PMFBY जैसी सरकारी योजनाओं के बारे में, या भूमि अधिकार, जल अधिकार, या न्यूनतम समर्थन मूल्य के बारे में पूछने का प्रयास करें।",
        'found_info': "आपके '{}' के बारे में पूछे गए प्रश्न से संबंधित कुछ जानकारी यहां दी गई है:"
    },
    'ta': {
        'no_results': "உங்கள் கேள்வி பற்றிய குறிப்பிட்ட தகவல்களை என்னால் கண்டுபிடிக்க முடியவில்லை. PM-KISAN, PMFBY போன்ற அரசு திட்டங்கள் அல்லது நில உரிமைகள், நீர் உரிமைகள் அல்லது குறைந்தபட்ச ஆதரவு விலைகள் பற்றி கேட்க முயற்சிக்கவும்.",
        'found_info': "உங்கள் '{}' பற்றிய கேள்விக்கு தொடர்புடைய சில தகவல்கள் இங்கே:"
    }
}

class FarmerAIAssistantDL:
    def __init__(self):
        self.schemes_df = pd.DataFrame(schemes_data)
        self.rights_df = pd.DataFrame(legal_rights_data)
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.translator = None
        try:
            self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")
        except:
            st.warning("Translation model not available. Using fallback translation.")
        
        self.schemes_embeddings = self._get_embeddings(self.schemes_df.apply(
            lambda row: f"{row['title']} {row['description']} {row['eligibility']} {row['benefits']}",
            axis=1
        ).tolist())
        
        self.rights_embeddings = self._get_embeddings(self.rights_df.apply(
            lambda row: f"{row['title']} {row['description']} {row['relevantLaws']} {row['applicability']}",
            axis=1
        ).tolist())

    def _get_embeddings(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = sum_embeddings / sum_mask
            embeddings.append(embedding[0].numpy())
        return np.array(embeddings)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def translate_text(self, text, source_lang, target_lang):
        if source_lang == target_lang:
            return text
        if self.translator:
            try:
                result = self.translator(text, src_lang=source_lang, tgt_lang=target_lang)
                return result[0]['translation_text']
            except:
                pass
        if text in translations.get(source_lang, {}):
            return translations.get(target_lang, {}).get(text, text)
        return text

    def detect_language(self, text):
        devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        tamil_pattern = re.compile(r'[\u0B80-\u0BFF]')
        if devanagari_pattern.search(text):
            return 'hi'
        elif tamil_pattern.search(text):
            return 'ta'
        else:
            return 'en'

    def _match_partial_query(self, query, text):
        """Check if query is a substring of text (case-insensitive)."""
        return query.lower() in text.lower()

    def process_query(self, query_text, language=None):
        if not language:
            language = self.detect_language(query_text)
        processing_text = query_text
        if language != 'en':
            pass  # Translation logic can be added here if needed
        
        # Get embedding for the query
        query_embedding = self._get_embeddings([processing_text])[0]
        scheme_similarities = [self._cosine_similarity(query_embedding, emb) for emb in self.schemes_embeddings]
        rights_similarities = [self._cosine_similarity(query_embedding, emb) for emb in self.rights_embeddings]
        
        # Lowered threshold for embeddings
        similarity_threshold = 0.3  # Adjusted for more flexibility
        
        # Lists to store relevant results
        relevant_schemes = []
        relevant_rights = []
        
        # Combine embedding similarity with partial string matching
        for i, (score, row) in enumerate(zip(scheme_similarities, self.schemes_df.itertuples())):
            scheme_text = f"{row.title} {row.description} {row.eligibility} {row.benefits}"
            if score > similarity_threshold or self._match_partial_query(query_text, scheme_text):
                relevant_schemes.append(self.schemes_df.iloc[i].to_dict())
        
        for i, (score, row) in enumerate(zip(rights_similarities, self.rights_df.itertuples())):
            rights_text = f"{row.title} {row.description} {row.relevantLaws} {row.applicability}"
            if score > similarity_threshold or self._match_partial_query(query_text, rights_text):
                relevant_rights.append(self.rights_df.iloc[i].to_dict())
        
        # Prepare response
        if relevant_schemes or relevant_rights:
            response_text = translations[language]['found_info'].format(query_text)
        else:
            response_text = translations[language]['no_results']
        
        return {
            "text": response_text,
            "relatedSchemes": relevant_schemes,
            "relatedRights": relevant_rights
        }

# Streamlit App
def main():
    st.title("Farmer AI Assistant")
    st.write("""
    Ask about government schemes (e.g., PM-KISAN, PMFBY) or legal rights (e.g., land ownership, water rights) in English, Hindi, or Tamil!
    Even partial queries like 'PM' or 'land' will return related results.
    """)

    # Initialize the assistant
    if 'assistant' not in st.session_state:
        with st.spinner("Initializing the AI Assistant..."):
            st.session_state.assistant = FarmerAIAssistantDL()

    # Sidebar for language selection
    language_options = {'English': 'en', 'Hindi': 'hi', 'Tamil': 'ta'}
    selected_language = st.sidebar.selectbox("Select Language", list(language_options.keys()))
    language_code = language_options[selected_language]

    # Input query
    query = st.text_input("Enter your query here (even a few letters work!)")
    
    if st.button("Get Answer"):
        if query:
            with st.spinner("Processing your query..."):
                response = st.session_state.assistant.process_query(query, language_code)
            
            # Display response
            st.subheader("Response")
            st.write(response['text'])
            
            if response['relatedSchemes']:
                st.subheader("Related Government Schemes")
                for scheme in response['relatedSchemes']:
                    with st.expander(f"{scheme['title']}"):
                        st.write(f"**Description**: {scheme['description']}")
                        st.write(f"**Eligibility**: {scheme['eligibility']}")
                        st.write(f"**Benefits**: {scheme['benefits']}")
                        st.write(f"**Application Process**: {scheme['applicationProcess']}")

            if response['relatedRights']:
                st.subheader("Related Legal Rights")
                for right in response['relatedRights']:
                    with st.expander(f"{right['title']}"):
                        st.write(f"**Description**: {right['description']}")
                        st.write(f"**Relevant Laws**: {right['relevantLaws']}")
                        st.write(f"**Applicability**: {right['applicability']}")

        else:
            st.warning("Please enter a query!")

    st.sidebar.write("Built with Streamlit for farmers' assistance")

if __name__ == "__main__":
    main()
